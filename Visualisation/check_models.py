def model_poser(model, radius=0.05):
    import numpy as np
    from psbody.mesh import Mesh, MeshViewer
    from psbody.mesh.sphere import Sphere
    mv = MeshViewer()
    k = 'i'
    while k != 'p':
        list_meshes = []
        for j in model.J_transformed:
            list_meshes.append(Sphere(center=np.array(j), radius=radius).to_mesh())
        list_meshes.append(Mesh(v=model.r, f=model.f))
        mv.set_dynamic_meshes(list_meshes)
        click = mv.get_mouseclick()
        cord = np.array([click['x'], click['y'], click['z']])
        row = np.argmin(np.sum((model.J_transformed.r - cord) ** 2, axis=1))
        mv.set_dynamic_meshes(
            [Mesh(v=model.r, f=model.f), Sphere(center=model.J_transformed.r[row, :], radius=radius).to_mesh()])
        k = mv.get_keypress()
        while k in ['a', 'd', 'w', 's', 'r', 'f', 'm', 'u', 'j', 'k', 'h', 'o', 'l']:
            print(model.J_transformed.r)

            if k == 'w':
                model.pose[row * 3] = model.pose[row * 3].r - 0.05 * np.pi
            elif k == 's':
                model.pose[row * 3] = model.pose[row * 3].r + 0.05 * np.pi
            elif k == 'd':
                model.pose[row * 3 + 1] = model.pose[row * 3 + 1].r + 0.05 * np.pi
            elif k == 'a':
                model.pose[row * 3 + 1] = model.pose[row * 3 + 1].r - 0.05 * np.pi
            elif k == 'r':
                model.pose[row * 3 + 2] = model.pose[row * 3 + 2].r - 0.05 * np.pi
            elif k == 'f':
                model.pose[row * 3 + 2] = model.pose[row * 3 + 2].r + 0.05 * np.pi
            elif k == 'm':
                model.pose[:] = 0
            elif k == 'u':
                model.trans[0] = model.trans[0].r - 0.1
            elif k == 'j':
                model.trans[0] = model.trans[0].r + 0.1
            elif k == 'k':
                model.trans[1] = model.trans[1].r - 0.1
            elif k == 'h':
                model.trans[1] = model.trans[1].r + 0.1
            elif k == 'o':
                model.trans[2] = model.trans[2].r - 0.1
            elif k == 'l':
                model.trans[2] = model.trans[2].r + 0.1
            mv.set_dynamic_meshes([Mesh(v=model.r, f=model.f)])
            k = mv.get_keypress()

from ch.star import STAR
verts = STAR(gender='female',num_betas=10)
model_poser(verts)
import pdb;pdb.set_trace()
from mesh import Mesh, MeshViewer
mesh = Mesh(v=verts.r, f=verts.f)
mv = MeshViewer()
mv.set_static_meshes([mesh])
import pdb;pdb.set_trace()


if False:
    from tf.star import STAR
    import numpy as np
    import tensorflow as tf

    star = STAR(gender='male',num_betas=10)

    pose = tf.constant(np.zeros((1,72)),dtype=tf.float32)
    trans = tf.constant(np.zeros((1,3)),dtype=tf.float32)
    betas = tf.constant(np.zeros((1,10)),dtype=tf.float32)
    verts = star(pose,betas,trans)

    from mesh import Mesh , MeshViewer
    mesh = Mesh(v=verts.numpy()[0],f=star.f)
    mv = MeshViewer()
    mv.set_static_meshes([mesh])
    import pdb;pdb.set_trace()

from pytorch.star import STAR
star = STAR(gender='male',num_betas=10)
import torch
import numpy as np
from torch.autograd import Variable
batch_size = 1
poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
poses = Variable(poses,requires_grad=True)
betas = torch.cuda.FloatTensor(np.zeros((batch_size,10)))
betas = Variable(betas,requires_grad=True)
trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
trans = Variable(trans,requires_grad=True)
v = star(poses,betas,trans)


from mesh import Mesh, MeshViewer
mesh = Mesh(v=v.cpu().detach().numpy()[0], f=star.f)
mv = MeshViewer()
mv.set_static_meshes([mesh])
import pdb;pdb.set_trace()
