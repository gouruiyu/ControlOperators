"""
A simple interactive animation renderer running a ML-based controller
Based on https://github.com/orangeduck/GenoViewPython/blob/main/genoview.py
"""

import struct
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cffi
from pyray import (
    Vector2, Vector3, Vector4, Transform, Matrix, Camera3D, 
    Color, Rectangle, Model, ModelAnimation, Mesh, BoneInfo, 
    Texture, RenderTexture)
from raylib import *
import quat
import control_operators as co
from control_encoder import ControlEncoderBase, NullControlEncoder, UberControlEncoder
from gameplay_input import GameplayInput
import networks
import bvh

ffi = cffi.FFI()

#----------------------------------------------------------------------------------
# Camera
#----------------------------------------------------------------------------------

class Camera:

    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()
    
    def update(
        self,
        target,
        azimuthDelta,
        altitudeDelta,
        offsetDeltaX,
        offsetDeltaY,
        mouseWheel,
        dt):

        self.azimuth = self.azimuth + 1.0 * dt * -azimuthDelta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitudeDelta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance +  20.0 * dt * -mouseWheel, 0.1, 100.0)
        
        rotationAzimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotationAzimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))

        rotationAltitude = QuaternionFromAxisAngle(axis, self.altitude)

        localOffset = Vector3(dt * offsetDeltaX, dt * -offsetDeltaY, 0.0)
        localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth)

        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(localOffset, rotationAltitude))

        cameraTarget = Vector3Add(self.offset, target)
        eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude))

        self.cam3d.target = cameraTarget
        self.cam3d.position = eye        


#----------------------------------------------------------------------------------
# Shadow Maps
#----------------------------------------------------------------------------------

class ShadowLight:
    
    def __init__(self):
        
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0
    
    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadShadowMap(target):
    
    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginShadowMap(target, shadowLight):
    
    BeginTextureMode(target)
    ClearBackground(WHITE)
    
    rlDrawRenderBatchActive()      # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width/2, shadowLight.width/2, 
        -shadowLight.height/2, shadowLight.height/2, 
        shadowLight.near, shadowLight.far)

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    # The float16 must be held in a variable: passing MatrixToFloatV(...).v directly
    # lets the temporary be freed before rlMultMatrixf reads it (renders nothing on macOS)
    matViewF = MatrixToFloatV(matView)
    rlMultMatrixf(matViewF.v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D    


def EndShadowMap():
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)     # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)      # Switch back to modelview matrix
    rlLoadIdentity()                # Reset current matrix (modelview)

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D

    EndTextureMode()


def LoadShaderCompat(vsPath, fsPath):
    # macOS's OpenGL driver lacks GL_ARB_ES3_compatibility, so '#version 300 es'
    # shaders fail to compile there. The same source is valid desktop GLSL 330
    # (precision qualifiers are legal no-ops), so swap the version directive.
    if sys.platform != 'darwin':
        return LoadShader(vsPath, fsPath)

    def patch(path):
        src = Path(path.decode()).read_text()
        return src.replace('#version 300 es', '#version 330', 1).encode()

    return LoadShaderFromMemory(patch(vsPath), patch(fsPath))


def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new('int*'); slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)


#----------------------------------------------------------------------------------
# GBuffer
#----------------------------------------------------------------------------------

class GBuffer:
    
    def __init__(self):
        self.id = 0              # OpenGL framebuffer object id
        self.color = Texture()   # Color buffer attachment texture 
        self.normal = Texture()  # Normal buffer attachment texture 
        self.depth = Texture()   # Depth buffer attachment texture


def LoadGBuffer(width, height):
    
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id
    
    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2) 

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    aspect = float(target.color.width)/float(target.color.height)

    # NOTE: zNear and zFar values are important when computing depth buffer values
    if camera.projection == CAMERA_PERSPECTIVE:

        # Setup perspective projection
        top = rlGetCullDistanceNear()*np.tan(camera.fovy*0.5*DEG2RAD)
        right = top*aspect

        rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    elif camera.projection == CAMERA_ORTHOGRAPHIC:

        # Setup orthographic projection
        top = camera.fovy/2.0
        right = top*aspect

        rlOrtho(-right, right, -top,top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(camera.position, camera.target, camera.up)
    # See BeginShadowMap: the float16 must outlive the rlMultMatrixf call
    matViewF = MatrixToFloatV(matView)
    rlMultMatrixf(matViewF.v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D


def EndGBuffer(windowWidth, windowHeight):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch
    
    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D
    rlActiveDrawBuffers(1) 
    rlDisableFramebuffer()          # Disable render target (fbo)

    # Restore the default framebuffer's viewport in render pixels; on HiDPI (macOS
    # retina) this is larger than the window size the GBuffer pass set above
    rlViewport(0, 0, GetRenderWidth(), GetRenderHeight())
    rlSetFramebufferWidth(GetRenderWidth())
    rlSetFramebufferHeight(GetRenderHeight())

    rlMatrixMode(RL_PROJECTION)         # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack
    rlLoadIdentity()                    # Reset current matrix (projection)
    rlOrtho(0, windowWidth, windowHeight, 0, 0.0, 1.0)

    rlMatrixMode(RL_MODELVIEW)          # Switch back to modelview matrix
    rlLoadIdentity()                    # Reset current matrix (modelview)


#----------------------------------------------------------------------------------
# Geno Character Mesh
#----------------------------------------------------------------------------------

def FileRead(out, size, f):
    ffi.memmove(out, f.read(size), size)


def LoadGenoModel(fileName: Path):
    
    MATERIAL_SIZE = ffi.sizeof(Mesh())
    MESH_SIZE = ffi.sizeof(Mesh())
    INT_SIZE = ffi.sizeof('int')
    FLOAT_SIZE = ffi.sizeof('float')
    BONEINFO_SIZE = ffi.sizeof(BoneInfo())
    TRANSFORM_SIZE = ffi.sizeof(Transform())
    MATRIX_SIZE = ffi.sizeof(Matrix())
    UCHAR_SIZE = ffi.sizeof('unsigned char')
    USHORT_SIZE = ffi.sizeof('unsigned short')
    
    model = Model()
    model.transform = MatrixIdentity()
  
    with open(fileName, "rb") as f:
        
        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * MATERIAL_SIZE)
        model.materials[0] = LoadMaterialDefault()

        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * INT_SIZE)
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * MESH_SIZE)
        model.meshes[0].vertexCount = struct.unpack('I', f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack('I', f.read(4))[0]
        model.boneCount = struct.unpack('I', f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * FLOAT_SIZE)
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * FLOAT_SIZE)
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * FLOAT_SIZE)
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * UCHAR_SIZE)
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * FLOAT_SIZE)
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * USHORT_SIZE)
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * FLOAT_SIZE)
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * FLOAT_SIZE)
        model.bones =  MemAlloc(model.boneCount * BONEINFO_SIZE)
        model.bindPose =  MemAlloc(model.boneCount * TRANSFORM_SIZE)
        
        FileRead(model.meshes[0].vertices, FLOAT_SIZE * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].texcoords, FLOAT_SIZE * model.meshes[0].vertexCount * 2, f)
        FileRead(model.meshes[0].normals, FLOAT_SIZE * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].boneIds, UCHAR_SIZE * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].boneWeights, FLOAT_SIZE * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].indices, USHORT_SIZE * model.meshes[0].triangleCount * 3, f)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, FLOAT_SIZE * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, FLOAT_SIZE * model.meshes[0].vertexCount * 3)
        FileRead(model.bones, BONEINFO_SIZE * model.boneCount, f)
        FileRead(model.bindPose, TRANSFORM_SIZE * model.boneCount, f)
        
        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * MATRIX_SIZE)
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()
    
    UploadMesh(ffi.addressof(model.meshes[0]), True)
    
    return model


def GetModelBindPoseAsNumpyArrays(model):
    
    bindPos = np.zeros([model.boneCount, 3])
    bindRot = np.zeros([model.boneCount, 4])
    
    for boneId in range(model.boneCount):
        bindTransform = model.bindPose[boneId]
        bindPos[boneId] = (bindTransform.translation.x, bindTransform.translation.y, bindTransform.translation.z)
        bindRot[boneId] = (bindTransform.rotation.w, bindTransform.rotation.x, bindTransform.rotation.y, bindTransform.rotation.z)
        
    return bindPos, bindRot
    
    
def UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot, animPos, animRot):
    
    meshPos = quat.mul_vec(animRot, quat.inv_mul_vec(bindRot, -bindPos)) + animPos
    meshRot = quat.mul_inv(animRot, bindRot)
    
    matArray = np.frombuffer(ffi.buffer(
        model.meshes[0].boneMatrices, model.boneCount * 4 * 4 * 4), 
        dtype=np.float32).reshape([model.boneCount, 4, 4])
    
    matArray[:,:3,:3] = quat.to_xform(meshRot)
    matArray[:,:3,3] = meshPos


# Computing fk using transformation matrices is much more efficient since it minimizes python overhead
def ForwardKinematics(locRot, locPos, parents):
    
    locXforms = np.zeros([len(locRot), 4, 4])
    locXforms[...,:3,:3] = quat.to_xform(locRot)
    locXforms[...,:3,3] = locPos
    locXforms[...,3,3] = 1
    
    gloXforms = locXforms.copy()
    for i in range(1, len(parents)):
        gloXforms[...,i,:,:] = gloXforms[...,parents[i],:,:] @ locXforms[...,i,:,:]
    
    gloRot = quat.from_xform(gloXforms)
    gloPos = gloXforms[...,:3,3]
    
    return gloRot, gloPos


# Adapted from:
# https://theorangeduck.com/page/simple-two-joint
def TwoBoneInverseKinematics(
    globalPelvis, 
    globalHip, 
    globalKnee, 
    globalHeel, 
    targetHeel, 
    sideVector,
    maxLengthBuffer = 0.005):
    
    targetClamp = targetHeel.copy()
    targetLength = np.sqrt(np.sum(np.square(targetHeel - globalHip[:3,3])))

    maxExtension = (
        np.sqrt(np.sum(np.square(globalHip[:3,3] - globalKnee[:3,3]))) + 
        np.sqrt(np.sum(np.square(globalKnee[:3,3] - globalHeel[:3,3]))) - 
        maxLengthBuffer)

    if targetLength > maxExtension:
        saturation = (1.0 - np.exp(-(targetLength - maxExtension) / maxLengthBuffer))
        targetClamp = globalHip[:3,3] + (maxExtension + maxLengthBuffer * saturation) * ((targetHeel - globalHip[:3,3]) / targetLength)
    
    axisDwn = globalHeel[:3,3] - globalHip[:3,3]
    axisDwn /= np.sqrt(np.sum(np.square(axisDwn)))
    
    axisFwd = np.cross(axisDwn, sideVector)
    axisFwd /= np.sqrt(np.sum(np.square(axisFwd)))
    
    axisRot = np.cross(axisDwn, axisFwd)
    axisRot /= np.sqrt(np.sum(np.square(axisRot)))

    a = globalHip[:3,3]
    b = globalKnee[:3,3]
    c = globalHeel[:3,3]
    t = targetClamp
    
    lab = np.sqrt(np.sum(np.square(b - a)))
    lcb = np.sqrt(np.sum(np.square(b - c)))
    lat = np.sqrt(np.sum(np.square(t - a)))
    lca = np.sqrt(np.sum(np.square(a - c)))

    acab0 = np.arccos(np.clip(np.dot((c - a) / lca, (b - a) / lab), -1.0, 1.0))
    babc0 = np.arccos(np.clip(np.dot((a - b) / lab, (c - b) / lcb), -1.0, 1.0))

    acab1 = np.arccos(np.clip((lab * lab + lat * lat - lcb * lcb) / (2.0 * lab * lat), -1.0, 1.0))
    babc1 = np.arccos(np.clip((lab * lab + lcb * lcb - lat * lat) / (2.0 * lab * lcb), -1.0, 1.0))

    r0 = quat.to_xform(quat.from_angle_axis(acab1 - acab0, axisRot))
    r1 = quat.to_xform(quat.from_angle_axis(babc1 - babc0, axisRot))
    r2 = quat.to_xform(quat.normalize(quat.between(globalHeel[:3,3] - globalHip[:3,3], targetClamp - globalHip[:3,3])))
    
    return (
        np.linalg.inv(globalPelvis[:3,:3]) @ r2 @ r0 @ globalHip[:3,:3],
        np.linalg.inv(globalHip[:3,:3]) @ r1 @ globalKnee[:3,:3])


class FootLockingState:
    pos = np.zeros([3])
    vel = np.zeros([3])
    input_pos = None
    input_vel = None
    offset_pos = np.zeros([3])
    offset_vel = np.zeros([3])
    time = 0.0
    contact = np.zeros([3])
    locked = False

# Adapted from:
# https://theorangeduck.com/page/creating-looping-animations-motion-capture
def InertializeCubicUpdate(pos, vel, time, input_pos, input_vel, offset_pos, offset_vel, dt, blendtime):
    
    t = np.clip((time + dt) / (blendtime + 1e-8), 0.0, 1.0)
    w0 = 2.0 * t * t * t - 3.0 * t * t + 1.0
    w1 = (t * t * t - 2.0 * t * t + t) * blendtime
    w2 = (6.0 * t * t - 6.0 * t) / (blendtime + 1e-8)
    w3 = 3.0 * t * t - 4.0 * t + 1.0

    return (
        input_pos + w0 * offset_pos + w1 * offset_vel,
        input_vel + w2 * offset_pos + w3 * offset_vel,
        time + dt)


def InertializeCubicTransition(offset_pos, offset_vel, time, src_pos, src_vel, dst_pos, dst_vel, blendtime):
    
    t = np.clip(time / (blendtime + 1e-8), 0.0, 1.0)
    w0 = 2.0 * t * t * t - 3.0 * t * t + 1.0
    w1 = (t * t * t - 2.0 * t * t + t) * blendtime
    w2 = (6.0 * t * t - 6.0 * t) / (blendtime + 1e-8)
    w3 = 3.0 * t * t - 4.0 * t + 1.0
    
    return (
        src_pos + w0 * offset_pos + w1 * offset_vel - dst_pos,
        src_vel + w2 * offset_pos + w3 * offset_vel - dst_vel,
        0.0)
    

def ApplyFootLockingAndInverseKinematics(
    locRot, 
    locPos, 
    leftState,
    rightState,
    leftContact,
    rightContact,
    boneNames,
    toeHeight,
    heelHeight,
    dt):
    
    # Convert to transformation matrices for efficiency
    
    locXforms = np.zeros([len(locRot), 4, 4])
    locXforms[...,:3,:3] = quat.to_xform(locRot)
    locXforms[...,:3,3] = locPos
    locXforms[...,3,3] = 1
    
    # Find Bone Indices
    
    leftToeEndIndex = boneNames.index('LeftToeBaseEnd')
    leftToeIndex = boneNames.index('LeftToeBase')
    leftHeelIndex = boneNames.index('LeftFoot')
    leftKneeIndex = boneNames.index('LeftLeg')
    leftHipIndex = boneNames.index('LeftUpLeg')
    
    rightToeEndIndex = boneNames.index('RightToeBaseEnd')
    rightToeIndex = boneNames.index('RightToeBase')
    rightHeelIndex = boneNames.index('RightFoot')
    rightKneeIndex = boneNames.index('RightLeg')
    rightHipIndex = boneNames.index('RightUpLeg')

    pelvisIndex = boneNames.index('Hips')
    rootIndex = boneNames.index('Simulation')
    
    # Compute Forward Kinematics down to toe
    
    globalPelvis = locXforms[rootIndex] @ locXforms[pelvisIndex]
    
    globalLeftHip = globalPelvis @ locXforms[leftHipIndex]
    globalLeftKnee = globalLeftHip @ locXforms[leftKneeIndex]
    globalLeftHeel = globalLeftKnee @ locXforms[leftHeelIndex]
    globalLeftToe = globalLeftHeel @ locXforms[leftToeIndex]
    
    globalRightHip = globalPelvis @ locXforms[rightHipIndex]
    globalRightKnee = globalRightHip @ locXforms[rightKneeIndex]
    globalRightHeel = globalRightKnee @ locXforms[rightHeelIndex]
    globalRightToe = globalRightHeel @ locXforms[rightToeIndex]
    
    # Perform foot locking on left toe
    
    unlockDistance = 0.2
    lockBlendTime = 0.2
    
    if leftState.input_pos is None:
        leftState.input_vel = np.zeros([3])
        leftState.input_pos = globalLeftToe[:3,3]
    else:
        leftState.input_vel = (globalLeftToe[:3,3] - leftState.input_pos) / dt
        leftState.input_pos = globalLeftToe[:3,3]
    
    leftState.pos, leftState.vel, leftState.time = InertializeCubicUpdate(
        leftState.pos, leftState.vel, leftState.time,
        leftState.contact if leftState.locked else leftState.input_pos,
        np.zeros([3]) if leftState.locked else leftState.input_vel,
        leftState.offset_pos, 
        leftState.offset_vel,
        dt,
        lockBlendTime)
    
    leftLockDistance = np.sqrt(np.sum(np.square(leftState.pos - leftState.input_pos)))
    
    if not leftState.locked and leftContact and leftLockDistance < unlockDistance:
        
        leftState.locked = True
        leftState.contact = leftState.input_pos.copy()
        leftState.contact[1] = toeHeight
        
        leftState.offset_pos, leftState.offset_vel, leftState.time = InertializeCubicTransition(
            leftState.offset_pos, leftState.offset_vel, leftState.time,
            leftState.input_pos,
            leftState.input_vel,
            leftState.contact,
            np.zeros([3]),
            lockBlendTime)
        
    elif leftState.locked and (not leftContact or leftLockDistance > 0.01):
        
        leftState.locked = False
        
        leftState.offset_pos, leftState.offset_vel, leftState.time = InertializeCubicTransition(
            leftState.offset_pos, leftState.offset_vel, leftState.time,
            leftState.contact,
            np.zeros([3]),
            leftState.input_pos,
            leftState.input_vel,
            lockBlendTime)
    
    leftToeTarget = leftState.pos.copy()
    leftToeTarget[1] = np.maximum(leftToeTarget[1], toeHeight)
    
    # Perform foot locking on right toe
    
    if rightState.input_pos is None:
        rightState.input_vel = np.zeros([3])
        rightState.input_pos = globalRightToe[:3,3]
    else:
        rightState.input_vel = (globalRightToe[:3,3] - rightState.input_pos) / dt
        rightState.input_pos = globalRightToe[:3,3]
        
    rightState.pos, rightState.vel, rightState.time = InertializeCubicUpdate(
        rightState.pos, rightState.vel, rightState.time,
        rightState.contact if rightState.locked else rightState.input_pos,
        np.zeros([3]) if rightState.locked else rightState.input_vel,
        rightState.offset_pos, 
        rightState.offset_vel,
        dt,
        lockBlendTime)
    
    rightLockDistance = np.sqrt(np.sum(np.square(rightState.pos - rightState.input_pos)))
    
    if not rightState.locked and rightContact and rightLockDistance < unlockDistance:
        
        rightState.locked = True
        rightState.contact = rightState.input_pos.copy()
        rightState.contact[1] = toeHeight
        
        rightState.offset_pos, rightState.offset_vel, rightState.time = InertializeCubicTransition(
            rightState.offset_pos, rightState.offset_vel, rightState.time,
            rightState.input_pos,
            rightState.input_vel,
            rightState.contact,
            np.zeros([3]),
            lockBlendTime)
        
    elif rightState.locked and (not rightContact or rightLockDistance > 0.01):
        
        rightState.locked = False
        
        rightState.offset_pos, rightState.offset_vel, rightState.time = InertializeCubicTransition(
            rightState.offset_pos, rightState.offset_vel, rightState.time,
            rightState.contact,
            np.zeros([3]),
            rightState.input_pos,
            rightState.input_vel,
            lockBlendTime)
    
    rightToeTarget = rightState.pos.copy()
    rightToeTarget[1] = np.maximum(rightToeTarget[1], toeHeight)
    
    # Apply small pelvis offset
    
    globalPelvis[:3,3] += np.array([0.0, -0.01, 0.0])
    
    locXforms[pelvisIndex] = np.linalg.inv(locXforms[rootIndex]) @ globalPelvis
    
    # Re-compute FK
    
    globalLeftHip = globalPelvis @ locXforms[leftHipIndex]
    globalLeftKnee = globalLeftHip @ locXforms[leftKneeIndex]
    globalLeftHeel = globalLeftKnee @ locXforms[leftHeelIndex]
    globalLeftToe = globalLeftHeel @ locXforms[leftToeIndex]
    
    globalRightHip = globalPelvis @ locXforms[rightHipIndex]
    globalRightKnee = globalRightHip @ locXforms[rightKneeIndex]
    globalRightHeel = globalRightKnee @ locXforms[rightHeelIndex]
    globalRightToe = globalRightHeel @ locXforms[rightToeIndex]
    
    # Find target heel position
    
    targetLeftHeel = leftToeTarget + (globalLeftHeel[:3,3] - globalLeftToe[:3,3])
    targetLeftHeel[1] = np.maximum(targetLeftHeel[1], heelHeight)
    
    targetRightHeel = rightToeTarget + (globalRightHeel[:3,3] - globalRightToe[:3,3])
    targetRightHeel[1] = np.maximum(targetRightHeel[1], heelHeight)
    
    # Solve two-joint IK for heel position
    
    leftKneeSide = globalLeftKnee[:3,:3] @ np.array([1.0, 0.0, 0.0])
    rightKneeSide = globalRightKnee[:3,:3] @ np.array([1.0, 0.0, 0.0])
    
    locXforms[leftHipIndex,:3,:3], locXforms[leftKneeIndex,:3,:3] = TwoBoneInverseKinematics(
        globalPelvis, 
        globalLeftHip, 
        globalLeftKnee, 
        globalLeftHeel, 
        targetLeftHeel, 
        leftKneeSide)
        
    locXforms[rightHipIndex,:3,:3], locXforms[rightKneeIndex,:3,:3] = TwoBoneInverseKinematics(
        globalPelvis, 
        globalRightHip, 
        globalRightKnee, 
        globalRightHeel, 
        targetRightHeel, 
        rightKneeSide)
        
    # Perform look-at for Heel to Toe
    
    globalLeftHip = globalPelvis @ locXforms[leftHipIndex]
    globalLeftKnee = globalLeftHip @ locXforms[leftKneeIndex]
    globalLeftHeel = globalLeftKnee @ locXforms[leftHeelIndex]
    globalLeftToe = globalLeftHeel @ locXforms[leftToeIndex]
    
    globalRightHip = globalPelvis @ locXforms[rightHipIndex]
    globalRightKnee = globalRightHip @ locXforms[rightKneeIndex]
    globalRightHeel = globalRightKnee @ locXforms[rightHeelIndex]
    globalRightToe = globalRightHeel @ locXforms[rightToeIndex]
    
    leftHeelRotation = (quat.to_xform(quat.normalize(quat.between(
        globalLeftToe[:3,3] - globalLeftHeel[:3,3],
        leftToeTarget - globalLeftHeel[:3,3]))) @ globalLeftHeel[:3,:3])

    rightHeelRotation = (quat.to_xform(quat.normalize(quat.between(
        globalRightToe[:3,3] - globalRightHeel[:3,3],
        rightToeTarget - globalRightHeel[:3,3]))) @ globalRightHeel[:3,:3])
    
    locXforms[leftHeelIndex,:3,:3] = np.linalg.inv(globalLeftKnee[:3,:3]) @ leftHeelRotation
    locXforms[rightHeelIndex,:3,:3] = np.linalg.inv(globalRightKnee[:3,:3]) @ rightHeelRotation
    
    # Perform look-at for Toe to Toe End
    
    globalLeftHeel = globalLeftKnee @ locXforms[leftHeelIndex]
    globalLeftToe = globalLeftHeel @ locXforms[leftToeIndex]
    globalLeftToeEnd = globalLeftToe @ locXforms[leftToeEndIndex]
    
    globalRightHeel = globalRightKnee @ locXforms[rightHeelIndex]
    globalRightToe = globalRightHeel @ locXforms[rightToeIndex]
    globalRightToeEnd = globalRightToe @ locXforms[rightToeEndIndex]
    
    leftToeEndTarget = globalLeftToeEnd[:3,3]
    leftToeEndTarget[1] = np.maximum(leftToeEndTarget[1], toeHeight)
    
    rightToeEndTarget = globalRightToeEnd[:3,3]
    rightToeEndTarget[1] = np.maximum(rightToeEndTarget[1], toeHeight)
    
    leftToeRotation = (quat.to_xform(quat.normalize(quat.between(
        globalLeftToeEnd[:3,3] - globalLeftToe[:3,3],
        leftToeEndTarget - globalLeftToe[:3,3]))) @ globalLeftToeEnd[:3,:3])
    
    rightToeRotation = (quat.to_xform(quat.normalize(quat.between(
        globalRightToeEnd[:3,3] - globalRightToe[:3,3],
        rightToeEndTarget - globalRightToe[:3,3]))) @ globalRightToeEnd[:3,:3])
    
    locXforms[leftToeIndex,:3,:3] = np.linalg.inv(globalLeftHeel[:3,:3]) @ leftToeRotation
    locXforms[rightToeIndex,:3,:3] = np.linalg.inv(globalRightHeel[:3,:3]) @ rightToeRotation
    
    # Convert back to quaternions and positions
    
    locRot, locPos = quat.from_xform(locXforms[...,:3,:3]), locXforms[...,:3,3]
    
    return locRot, locPos
    

def main():
    
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    DATA_ROOT = Path(__file__).resolve().parent / "data" / "lafan1_resolved"
    RESOURCES_ROOT = Path(__file__).resolve().parent / "resources"
    
    def RES(p: str) -> bytes:
        """Helper function to get resource path."""
        return str((RESOURCES_ROOT / p).resolve()).encode()
    
    # Setup
    
    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"Character Controller - DEBUG")
    SetTargetFPS(60)
    rlSetClipPlanes(0.01, 50.0)
    
    basicShader = LoadShaderCompat(RES("basic.vs"), RES("basic.fs"))
    basicShaderSpecularity = GetShaderLocation(basicShader, b"specularity")
    basicShaderGlossiness = GetShaderLocation(basicShader, b"glossiness")
    basicShaderCamClipNear = GetShaderLocation(basicShader, b"camClipNear")
    basicShaderCamClipFar = GetShaderLocation(basicShader, b"camClipFar")
    
    skinnedBasicShader = LoadShaderCompat(RES("skinnedBasic.vs"), RES("basic.fs"))
    skinnedBasicShaderSpecularity = GetShaderLocation(skinnedBasicShader, b"specularity")
    skinnedBasicShaderGlossiness = GetShaderLocation(skinnedBasicShader, b"glossiness")
    skinnedBasicShaderCamClipNear = GetShaderLocation(skinnedBasicShader, b"camClipNear")
    skinnedBasicShaderCamClipFar = GetShaderLocation(skinnedBasicShader, b"camClipFar")
    
    shadowShader = LoadShaderCompat(RES("shadow.vs"), RES("shadow.fs"))
    shadowShaderLightClipNear = GetShaderLocation(shadowShader, b"lightClipNear")
    shadowShaderLightClipFar = GetShaderLocation(shadowShader, b"lightClipFar")
    
    skinnedShadowShader = LoadShaderCompat(RES("skinnedShadow.vs"), RES("shadow.fs"))
    skinnedShadowShaderLightClipNear = GetShaderLocation(skinnedShadowShader, b"lightClipNear")
    skinnedShadowShaderLightClipFar = GetShaderLocation(skinnedShadowShader, b"lightClipFar")
    
    ssaoShader = LoadShaderCompat(RES("post.vs"), RES("ssao.fs"))
    ssaoShaderGBufferNormal = GetShaderLocation(ssaoShader, b"gbufferNormal")
    ssaoShaderGBufferDepth = GetShaderLocation(ssaoShader, b"gbufferDepth")
    ssaoShaderCamView = GetShaderLocation(ssaoShader, b"camView")
    ssaoShaderCamProj = GetShaderLocation(ssaoShader, b"camProj")
    ssaoShaderCamInvProj = GetShaderLocation(ssaoShader, b"camInvProj")
    ssaoShaderCamInvViewProj = GetShaderLocation(ssaoShader, b"camInvViewProj")
    ssaoShaderLightViewProj = GetShaderLocation(ssaoShader, b"lightViewProj")
    ssaoShaderShadowMap = GetShaderLocation(ssaoShader, b"shadowMap")
    ssaoShaderShadowInvResolution = GetShaderLocation(ssaoShader, b"shadowInvResolution")
    ssaoShaderCamClipNear = GetShaderLocation(ssaoShader, b"camClipNear")
    ssaoShaderCamClipFar = GetShaderLocation(ssaoShader, b"camClipFar")
    ssaoShaderLightClipNear = GetShaderLocation(ssaoShader, b"lightClipNear")
    ssaoShaderLightClipFar = GetShaderLocation(ssaoShader, b"lightClipFar")
    ssaoShaderLightDir = GetShaderLocation(ssaoShader, b"lightDir")
    
    blurShader = LoadShaderCompat(RES("post.vs"), RES("blur.fs"))
    blurShaderGBufferNormal = GetShaderLocation(blurShader, b"gbufferNormal")
    blurShaderGBufferDepth = GetShaderLocation(blurShader, b"gbufferDepth")
    blurShaderInputTexture = GetShaderLocation(blurShader, b"inputTexture")
    blurShaderCamInvProj = GetShaderLocation(blurShader, b"camInvProj")
    blurShaderCamClipNear = GetShaderLocation(blurShader, b"camClipNear")
    blurShaderCamClipFar = GetShaderLocation(blurShader, b"camClipFar")
    blurShaderInvTextureResolution = GetShaderLocation(blurShader, b"invTextureResolution")
    blurShaderBlurDirection = GetShaderLocation(blurShader, b"blurDirection")

    lightingShader = LoadShaderCompat(RES("post.vs"), RES("lighting.fs"))
    lightingShaderGBufferColor = GetShaderLocation(lightingShader, b"gbufferColor")
    lightingShaderGBufferNormal = GetShaderLocation(lightingShader, b"gbufferNormal")
    lightingShaderGBufferDepth = GetShaderLocation(lightingShader, b"gbufferDepth")
    lightingShaderSSAO = GetShaderLocation(lightingShader, b"ssao")
    lightingShaderCamPos = GetShaderLocation(lightingShader, b"camPos")
    lightingShaderCamInvViewProj = GetShaderLocation(lightingShader, b"camInvViewProj")
    lightingShaderLightDir = GetShaderLocation(lightingShader, b"lightDir")
    lightingShaderSunColor = GetShaderLocation(lightingShader, b"sunColor")
    lightingShaderSunStrength = GetShaderLocation(lightingShader, b"sunStrength")
    lightingShaderSkyColor = GetShaderLocation(lightingShader, b"skyColor")
    lightingShaderSkyStrength = GetShaderLocation(lightingShader, b"skyStrength")
    lightingShaderGroundStrength = GetShaderLocation(lightingShader, b"groundStrength")
    lightingShaderAmbientStrength = GetShaderLocation(lightingShader, b"ambientStrength")
    lightingShaderExposure = GetShaderLocation(lightingShader, b"exposure")
    lightingShaderCamClipNear = GetShaderLocation(lightingShader, b"camClipNear")
    lightingShaderCamClipFar = GetShaderLocation(lightingShader, b"camClipFar")

    fxaaShader = LoadShaderCompat(RES("post.vs"), RES("fxaa.fs"))
    fxaaShaderInputTexture = GetShaderLocation(fxaaShader, b"inputTexture")
    fxaaShaderInvTextureResolution = GetShaderLocation(fxaaShader, b"invTextureResolution")
    
    camera = Camera()
    
    # Directional light
    lightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
    
    # Shadow
    shadowLight = ShadowLight()
    shadowLight.target = Vector3Zero()
    shadowLight.position = Vector3Scale(lightDir, -5.0)
    shadowLight.up = Vector3(0.0, 1.0, 0.0)
    shadowLight.width = 5.0
    shadowLight.height = 5.0
    shadowLight.near = 0.01
    shadowLight.far = 10.0

    shadowWidth = 1024
    shadowHeight = 1024
    shadowInvResolution = Vector2(1.0 / shadowWidth, 1.0 / shadowHeight)
    shadowMap = LoadShadowMap(shadowWidth, shadowHeight)
    
    # GBuffer and render textures
    gbuffer = LoadGBuffer(SCREEN_WIDTH, SCREEN_HEIGHT)
    lighted = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT)
    ssaoFront = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT)
    ssaoBack = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Ground plane
    groundMesh = GenMeshPlane(20.0, 20.0, 10, 10)
    groundModel = LoadModelFromMesh(groundMesh)
    groundPosition = Vector3(0.0, -0.01, 0.0)
    
    # Load gamepad texture
    xboxTexture = LoadTexture(RES("xbox.png"))
    
    # Geno character model
    genoModel = LoadGenoModel(RESOURCES_ROOT / "Geno.bin")
    genoPosition = Vector3(0.0, 0.0, 0.0)
    bindPos, bindRot = GetModelBindPoseAsNumpyArrays(genoModel)
    
    # Animation data
    bvhData = bvh.load(str(RESOURCES_ROOT / "Geno_bind.bvh"))
    parents = bvhData['parents']
    names = bvhData['names']
    nbones = len(parents)
    localPositions = 0.01 * bvhData['positions'].copy().astype(np.float32)
    localRotations = quat.unroll(quat.from_euler(np.radians(bvhData['rotations']), order=bvhData['order']))
    globalRotations, globalPositions = quat.fk(localRotations, localPositions, parents)
    
    toeHeight = globalPositions[0,names.index('LeftToeBase'),1]
    heelHeight = globalPositions[0,names.index('LeftFoot'),1]
    
    leftFootLockingState = FootLockingState()
    rightFootLockingState = FootLockingState()
    
    # Controller initialization
    print("\n" + "="*80)
    print("CONTROLLER INITIALIZATION")
    print("="*80)

    autoencoder_path = DATA_ROOT / "autoencoder.ptz"
    controller_path = DATA_ROOT / "UberControlEncoder" / "controller.ptz"
    # controller_path = DATA_ROOT / "NullControlEncoder" / "controller.ptz"
    database_path = DATA_ROOT / "database.npz"
    xnpz_path = DATA_ROOT / "X.npz"
    znpz_path = DATA_ROOT / "Z.npz"
    
    use_fm = False
    
    if not all(p.exists() for p in [autoencoder_path, controller_path, database_path, xnpz_path, znpz_path]):
        print("Missing controller files, running in BIND POSE mode")
        use_fm = False
    else:
        try:
            # Load data
            xdata = np.load(xnpz_path, allow_pickle=True)
            Xoffset = torch.as_tensor(xdata['Xoffset'], device='cpu', dtype=torch.float32)
            Xscale = torch.as_tensor(xdata['Xscale'], device='cpu', dtype=torch.float32)
            Xdist = torch.as_tensor(xdata['Xdist'], device='cpu', dtype=torch.float32)
            Xref_pos = xdata['Xref_pos']
            X_dim = len(Xoffset)
            
            zdata = np.load(znpz_path, allow_pickle=True)
            Z = zdata['Z']
            Zoffset = torch.as_tensor(zdata['Zoffset'], device='cpu', dtype=torch.float32)
            Zscale = torch.as_tensor(zdata['Zscale'], device='cpu', dtype=torch.float32)
            Zdist = torch.as_tensor(zdata['Zdist'], device='cpu', dtype=torch.float32)
            Zmin = torch.as_tensor(zdata['Zmin'], device='cpu', dtype=torch.float32)
            Zmax = torch.as_tensor(zdata['Zmax'], device='cpu', dtype=torch.float32)
            Z_dim = Z.shape[1]
            
            database_data = np.load(database_path, allow_pickle=True)
            nbones_database = database_data['positions'].shape[1]
            database_parents = database_data['parents']
            database_names = database_data['names']
            
            encoder_network = networks.MLP(inp=X_dim, out=256, hidden=512, depth=2)
            decoder_network = networks.MLP(inp=256, out=X_dim, hidden=512, depth=2)

            control_encoder: ControlEncoderBase = UberControlEncoder()
            # control_encoder: ControlEncoderBase = NullControlEncoder()

            # build flow model 
            denoiser_network = networks.SkipCatMLP(inp=(Z.shape[1]*2 + control_encoder.output_size() + 1), out=Z.shape[1], hidden=1024, depth=10)
            
            autoencoder_data = torch.load(autoencoder_path, weights_only=True, map_location='cpu')
            encoder_network.load_state_dict(autoencoder_data['encoder'])
            decoder_network.load_state_dict(autoencoder_data['decoder'])
            
            # Load Null controller and denoiser
            controller_data = torch.load(controller_path, weights_only=True, map_location='cpu')
            control_encoder.root.load_state_dict(controller_data['control_encoder'])
            denoiser_network.load_state_dict(controller_data['denoiser'])
            print(f"Loaded controller from {controller_path}")
            
            encoder_network.eval()
            decoder_network.eval()
            control_encoder.eval()
            denoiser_network.eval()
            
            # Helpers

            def reset_animation():
                start_frame = np.random.randint(0, len(Z))
                Zprev_new = torch.as_tensor(Z[start_frame][None], device='cpu', dtype=torch.float32)
                print(f"Reset animation - new start frame: {start_frame}")
                return Zprev_new, start_frame

            # TODO: refactor these same with train.py inference
            @torch.no_grad()
            @torch.jit.script
            def inference_cpu(Zprev, Zdist, Zmin, Zmax, Cnext, S : int = 4):
                Znext = Zdist * torch.randn_like(Zprev, device='cpu')
                for s in range(S):
                    t = torch.full([Zprev.shape[0], 1], s / S, device='cpu', dtype=torch.float32)
                    denoiser_input = torch.cat([Znext, Zprev, Cnext, t], dim=1)
                    Znext = Znext + (1 / S) * Zdist * denoiser_network(denoiser_input)
                return Znext.clip(Zmin, Zmax)
            
            @torch.no_grad()
            def decode_pose(Zinput, Xroot_pos, Xroot_rot, dt):
                
                # Decode to normalized pose vector
                Xpose = (Xdist * decoder_network(Zinput * Zscale + Zoffset)).cpu().numpy()[0]
                
                # Denormalize pose vector
                Xpose = Xpose * Xscale.numpy() + Xoffset.numpy()
                
                # Unpack pose
                Xrvel = Xpose[0:3]
                Xrang = Xpose[3:6]
                Xhip  = Xpose[6:9]
                Xrot  = quat.from_xform_xy(Xpose[9:9+(nbones_database-1)*6].reshape([nbones_database - 1, 3, 2]))
                Xhvel = Xpose[9+(nbones_database-1)*6:12+(nbones_database-1)*6]
                Xang  = Xpose[12+(nbones_database-1)*6:12+(nbones_database-1)*9].reshape([nbones_database - 1, 3])
                leftContact = Xpose[12+(nbones_database-1)*9+0:12+(nbones_database-1)*9+1] > 0.5
                rightContact = Xpose[12+(nbones_database-1)*9+1:12+(nbones_database-1)*9+2] > 0.5
                assert len(Xpose) == 12+(nbones_database-1)*9+2
            
                # Integrate root motion
                Xroot_vel_world = quat.mul_vec(Xroot_rot, Xrvel)
                Xroot_ang_world = quat.mul_vec(Xroot_rot, Xrang)
                Xroot_pos_new = (dt * Xroot_vel_world) + Xroot_pos
                Xroot_rot_new = quat.mul(quat.from_scaled_angle_axis(dt * Xroot_ang_world), Xroot_rot)
                
                # Construct full pose state
                localPositions = np.concatenate([Xroot_pos_new[None], Xhip[None], Xref_pos[2:]], axis=0)
                localRotations = np.concatenate([Xroot_rot_new[None], Xrot], axis=0)
                localVelocities = np.concatenate([Xroot_vel_world[None], Xhvel[None], np.zeros([nbones_database - 2, 3])], axis=0)
                localAngularVelocities = np.concatenate([Xroot_ang_world[None], Xang], axis=0)
                
                return localPositions, localRotations, localVelocities, localAngularVelocities, leftContact, rightContact
        
            # Initialize animation from random starting frame
            Zprev, start_frame = reset_animation()
            
            use_fm = True
            print("Controller initialized")
            print(f"  - Device: CPU")
            print(f"  - Bones: {nbones}")
            print(f"  - Latent dim: {Z_dim}")
            print(f"  - Pose dim: {X_dim}")
            print(f"  - Starting from frame: {start_frame}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"ERROR loading neural network: {e}")
            import traceback
            traceback.print_exc()
            print("\nRunning in BIND POSE mode")
            print("="*80 + "\n")
            use_fm = False
            
    
    mode_label = "NEURAL NETWORK" if use_fm else "BIND POSE DEBUG"
    print(f"Starting render loop in {mode_label} mode...")
    
    frame_count = 0
    
    # Fixed frame rate
    dt = 1.0 / 60.0

    # Initialize root motion and simulation state
    Xroot_pos = np.zeros(3, dtype=np.float32)
    Xroot_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    localPositions, localRotations, localVelocities, localAngularVelocities, _, _ = decode_pose(Zprev, Xroot_pos, Xroot_rot, dt)

    # Initialize input
    gameplay_input = GameplayInput(
        control_type='uncontrolled'
    )
    
    # Pose Smoothing
    poseSmoothingPtr = ffi.new('float*'); poseSmoothingPtr[0] = 0.75
    
    while not WindowShouldClose():
        
        # Controller update
        if use_fm:
            try:
                # Update gameplay input from devices and sim states
                gameplay_input.dt = dt
                gameplay_input.update_from_gamepad(gamepad_id=0, deadzone=0.2)
                gameplay_input.update_from_keyboard()
                
                # Handle control mode switching
                if gameplay_input.apply_mode_switch():
                    print(f"Switched to {gameplay_input.control_type.upper()} mode")
                
                # Reset animation
                if gameplay_input.reset_requested:
                    Zprev, _ = reset_animation()
                    Xroot_pos = np.zeros(3, dtype=np.float32)
                    Xroot_rot = np.array([1, 0, 0, 0], dtype=np.float32)
                    localPositions, localRotations, localVelocities, localAngularVelocities, _, _ = decode_pose(Zprev, Xroot_pos, Xroot_rot, dt)
                    leftFootLockingState = FootLockingState()
                    rightFootLockingState = FootLockingState()
                    frame_count = 0
                    print("Animation reset")
                    gameplay_input.reset_requested = False
                
                gameplay_input.update_camera_state(
                    azimuth=camera.azimuth,
                    altitude=camera.altitude,
                    distance=camera.distance
                )
                
                gameplay_input.update_simulation_state(
                    position=Xroot_pos,
                    rotation=Xroot_rot,
                    velocity=localVelocities[0],
                    angular_velocity=localAngularVelocities[0]
                )
                
                # Generate runtime control
                Vnext = control_encoder.runtime_controls(gameplay_input)
                
                # Compute encoded control vector
                Cnext = control_encoder([Vnext])

                # Run inference to update Zprev
                Zprev = inference_cpu(Zprev, Zdist, Zmin, Zmax, Cnext, S=4)
                
                # Decode new pose and apply pose smoothing
                prevLocalPositions, prevLocalRotations = localPositions, localRotations
                (localPositions, localRotations, 
                 localVelocities, localAngularVelocities, 
                 leftContact, rightContact) = decode_pose(Zprev, Xroot_pos, Xroot_rot, dt)
                
                alpha = poseSmoothingPtr[0]
                localPositions = (1.0 - alpha) * localPositions + alpha * (dt * localVelocities + prevLocalPositions)
                localRotations = quat.nlerp_shortest(localRotations, quat.mul(quat.from_scaled_angle_axis(dt * localAngularVelocities), prevLocalRotations), alpha)
                
                # Update root 
                Xroot_pos = localPositions[0].copy()
                Xroot_rot = localRotations[0].copy()
            
                # Perform foot locking and IK

                modifiedRotations, modifiedPositions = ApplyFootLockingAndInverseKinematics(
                    localRotations, localPositions,
                    leftFootLockingState,
                    rightFootLockingState,
                    leftContact,
                    rightContact,
                    database_names.tolist(), toeHeight, heelHeight, dt)
            
                # Compute FK and update transforms
                globalRotations, globalPositions = ForwardKinematics(modifiedRotations, modifiedPositions, database_parents)
                UpdateModelPoseFromNumpyArrays(genoModel, bindPos, bindRot, globalPositions[1:], globalRotations[1:])
                
            except Exception as e:
                print(f"Error in model inference: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to bind pose
                use_fm = False
        
        hipPosition = Vector3(*globalPositions[0])
        shadowLight.target = Vector3(hipPosition.x, 0.0, hipPosition.z)
        shadowLight.position = Vector3Add(shadowLight.target, Vector3Scale(lightDir, -5.0))

        camera.update(
            Vector3(hipPosition.x, 0.75, hipPosition.z),
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseWheelMove(),
            GetFrameTime())
        
        rlDisableColorBlend()
        BeginDrawing()
        

        BeginShadowMap(shadowMap, shadowLight)
        
        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightClipNear = rlGetCullDistanceNear()
        lightClipFar = rlGetCullDistanceFar()

        lightClipNearPtr = ffi.new("float*"); lightClipNearPtr[0] = lightClipNear
        lightClipFarPtr = ffi.new("float*"); lightClipFarPtr[0] = lightClipFar
        
        SetShaderValue(shadowShader, shadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shadowShader, shadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        groundModel.materials[0].shader = shadowShader
        DrawModel(groundModel, groundPosition, 1.0, WHITE)
        
        genoModel.materials[0].shader = skinnedShadowShader
        DrawModel(genoModel, genoPosition, 1.0, WHITE)
        
        EndShadowMap()
        
        
        BeginGBuffer(gbuffer, camera.cam3d)
        
        camView = rlGetMatrixModelview()
        camProj = rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camClipNear = rlGetCullDistanceNear()
        camClipFar = rlGetCullDistanceFar()

        camClipNearPtr = ffi.new("float*"); camClipNearPtr[0] = camClipNear
        camClipFarPtr = ffi.new("float*"); camClipFarPtr[0] = camClipFar

        specularityPtr = ffi.new('float*'); specularityPtr[0] = 0.5
        glossinessPtr = ffi.new('float*'); glossinessPtr[0] = 10.0
        
        SetShaderValue(basicShader, basicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        groundModel.materials[0].shader = basicShader
        DrawModel(groundModel, groundPosition, 1.0, Color(190, 190, 190, 255))
        
        genoModel.materials[0].shader = skinnedBasicShader
        DrawModel(genoModel, genoPosition, 1.0, ORANGE)

        EndGBuffer(SCREEN_WIDTH, SCREEN_HEIGHT)
        

        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(ssaoShader)
        
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferDepth, gbuffer.depth)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamView, camView)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamProj, camProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvProj, camInvProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvViewProj, camInvViewProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderLightViewProj, lightViewProj)
        SetShaderValueShadowMap(ssaoShader, ssaoShaderShadowMap, shadowMap)
        SetShaderValue(ssaoShader, ssaoShaderShadowInvResolution, ffi.addressof(shadowInvResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(ssaoShader, ssaoShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        
        ClearBackground(WHITE)
        
        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0.0, 0.0),
            WHITE)

        EndShaderMode()
        EndTextureMode()
        

        BeginTextureMode(ssaoBack)
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(1.0, 0.0)
        blurInvTextureResolution = Vector2(1.0 / ssaoFront.texture.width, 1.0 / ssaoFront.texture.height)
        
        SetShaderValueTexture(blurShader, blurShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(blurShader, blurShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoFront.texture)
        SetShaderValueMatrix(blurShader, blurShaderCamInvProj, camInvProj)
        SetShaderValue(blurShader, blurShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderInvTextureResolution, ffi.addressof(blurInvTextureResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(ssaoBack.texture, Rectangle(0, 0, ssaoBack.texture.width, -ssaoBack.texture.height), Vector2(0, 0), WHITE)

        EndShaderMode()
        EndTextureMode()
        

        BeginTextureMode(ssaoFront)
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(0.0, 1.0)
        
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoBack.texture)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(ssaoFront.texture, Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height), Vector2(0, 0), WHITE)

        EndShaderMode()
        EndTextureMode()
        

        BeginTextureMode(lighted)
        BeginShaderMode(lightingShader)
        
        sunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        sunStrengthPtr = ffi.new('float*'); sunStrengthPtr[0] = 0.25
        skyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        skyStrengthPtr = ffi.new('float*'); skyStrengthPtr[0] = 0.15
        groundStrengthPtr = ffi.new('float*'); groundStrengthPtr[0] = 0.1
        ambientStrengthPtr = ffi.new('float*'); ambientStrengthPtr[0] = 1.0
        exposurePtr = ffi.new('float*'); exposurePtr[0] = 0.9
        
        SetShaderValueTexture(lightingShader, lightingShaderGBufferColor, gbuffer.color)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(lightingShader, lightingShaderSSAO, ssaoFront.texture)
        SetShaderValue(lightingShader, lightingShaderCamPos, ffi.addressof(camera.cam3d.position), SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(lightingShader, lightingShaderCamInvViewProj, camInvViewProj)
        SetShaderValue(lightingShader, lightingShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunColor, ffi.addressof(sunColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunStrength, sunStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderSkyColor, ffi.addressof(skyColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSkyStrength, skyStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderGroundStrength, groundStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderAmbientStrength, ambientStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderExposure, exposurePtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        ClearBackground(RAYWHITE)
        DrawTextureRec(gbuffer.color, Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height), Vector2(0, 0), WHITE)
        
        EndShaderMode()
        
        if use_fm:
            BeginMode3D(camera.cam3d)
            gameplay_input.draw_debug_visuals()
            EndMode3D()
        
        EndTextureMode()
        

        BeginShaderMode(fxaaShader)

        fxaaInvTextureResolution = Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
        
        SetShaderValueTexture(fxaaShader, fxaaShaderInputTexture, lighted.texture)
        SetShaderValue(fxaaShader, fxaaShaderInvTextureResolution, ffi.addressof(fxaaInvTextureResolution), SHADER_UNIFORM_VEC2)
        
        DrawTextureRec(lighted.texture, Rectangle(0, 0, lighted.texture.width, -lighted.texture.height), Vector2(0, 0), WHITE)
        
        EndShaderMode()
        
        # UI
        rlEnableColorBlend()
        
        DrawFPS(10, 10)
        
        if use_fm:
            DrawText(f"MODE: Flow Matching Controller".encode(), 10, 30, 20, BLUE)
            DrawText(f"Frame: {frame_count}".encode(), 10, 55, 20, BLACK)
        else:
            DrawText(b"MODE: Bind Pose", 10, 30, 20, RED)
            DrawText(f"Frame: {frame_count}".encode(), 10, 55, 20, BLACK)
            DrawText(b"Status: Static bind pose", 10, 80, 16, DARKGRAY)
        
        DrawText(b"Camera Controls:", 10, 130, 16, BLACK)
        DrawText(b"  Ctrl + Left Click: Rotate", 10, 150, 14, DARKGRAY)
        DrawText(b"  Ctrl + Right Click: Pan", 10, 170, 14, DARKGRAY)
        DrawText(b"  Mouse Wheel: Zoom", 10, 190, 14, DARKGRAY)

        DrawText(b"Control Mode:", 10, 220, 16, BLACK)
        modes = [
            (1, "Uncontrolled", "uncontrolled"),
            (2, "Trajectory", "trajectory"),
            (3, "Velocity Facing", "velocity_facing"),
        ]
        y = 240
        for num, label, ctype in modes:
            color = BLUE if use_fm and gameplay_input.control_type == ctype else DARKGRAY
            DrawText(f"  {num}: {label}".encode(), 10, y, 14, color)
            y += 20

        DrawText(b"  R: Reset", 10, y + 80, 14, DARKGRAY)
        DrawText(b"  Left Stick: Move", 10, y + 40, 14, DARKGRAY)
        if use_fm and gameplay_input.control_type == 'velocity_facing':
            DrawText(b"  Right Stick: Face Direction", 10, y + 60, 14, DARKGRAY)
        
        GuiSliderBar(Rectangle(SCREEN_WIDTH - 170, 20, 100, 20), b"Pose Smoothing", b"%5.2f" % poseSmoothingPtr[0], poseSmoothingPtr, 0.0, 1.0)

        gameplay_input.draw_joystick_debug(gamepad_id=0, texture=xboxTexture, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT)

        EndDrawing()
        frame_count += 1
    
    # Cleanup
    
    UnloadRenderTexture(lighted)
    UnloadRenderTexture(ssaoBack)
    UnloadRenderTexture(ssaoFront)
    UnloadGBuffer(gbuffer)
    UnloadShadowMap(shadowMap)

    UnloadTexture(xboxTexture)
    UnloadModel(genoModel)
    UnloadModel(groundModel)

    UnloadShader(fxaaShader)
    UnloadShader(blurShader)
    UnloadShader(ssaoShader)
    UnloadShader(lightingShader)
    UnloadShader(basicShader)
    UnloadShader(skinnedBasicShader)
    UnloadShader(skinnedShadowShader)
    UnloadShader(shadowShader)
    
    CloseWindow()

if __name__ == "__main__":
    main()
