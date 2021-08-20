from scipy.spatial.transform import Rotation

# Create a rotation object from Euler angles specifying axes of rotation
rot = Rotation.from_euler('xyz', [0.35, 0, 0.23], degrees=False)

# Convert to quaternions and print
rot_quat = rot.as_quat()
print(rot_quat)

#print(rot.as_euler('xyz', degrees=True))

rot = Rotation.from_quat(rot_quat)

# Convert the rotation to Euler angles given the axes of rotation
print(rot.as_euler('xyz', degrees=False))