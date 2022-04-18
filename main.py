from cmath import cos
from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
scene.set_background_color((0.3, 0.4, 0.6))


@ti.kernel
def initialize_voxels():
    n = 60
    r = 20
    p_center = vec3(n // 2, n // 2, n // 2)

    vec_face_to = vec3(0, 0, 1)
    vec_z = vec3(0, 1, 0)
    vec_normal = normalize(cross(vec_face_to, vec_z))

    mouse_angle = pi / 4
    mouse_angle_cos = ti.cos(mouse_angle)
    skin_thickness = 1

    # scene.set_voxel(p_center, 2, vec3(1, 1, 1))

    for i, j, k in ti.ndrange((-n, n), (-n, n), (-n, n)):
        x = ivec3(i, j, k)

        # surface
        if distance(x, p_center) < r + skin_thickness and distance(x,
                                                                   p_center) > r - skin_thickness:

            color = vec3(2, 2, 2)
            # mouse
            # project to the plane
            vec_mouse = vec3(i, j, k) - p_center
            vec_mouse_projected = vec_mouse - vec_normal * dot(vec_mouse, vec_normal)
            print(vec_mouse_projected)

            # angle to face to
            angle = dot(vec_mouse_projected,
                        vec_face_to) / (vec_mouse_projected.norm() * vec_face_to.norm())

            if angle <= mouse_angle_cos:
                color = vec3(1, 1, 0.3)

            if any(color != vec3(2, 2, 2)):
                scene.set_voxel(vec3(i, j, k), 1, color)


initialize_voxels()

scene.finish()
