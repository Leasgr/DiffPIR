"""
Compatibilite mpi4py optionnelle.

guided_diffusion/dist_util.py et guided_diffusion/image_datasets.py
utilisent `from mpi4py import MPI` pour repartir les donnees entre
processus lors d'un entrainement multi-GPU/multi-noeud. Or mpi4py est
volontairement commente dans requirements.txt de ce depot (installation
delicate sous Windows) et n'est donc pas disponible dans l'environnement
par defaut.

Pour permettre un entrainement/fine-tuning mono-GPU (ou CPU) SANS installer
MPI, ce module expose un objet `MPI` :
  - le vrai module mpi4py.MPI s'il est installe ET fonctionnel (entrainement
    multi-GPU/multi-noeud via `mpiexec -n N python scripts/image_train.py`),
  - sinon un mini-shim `MPI.COMM_WORLD` simulant un seul processus
    (rank=0, world_size=1, bcast=identite), suffisant pour
    dist_util.setup_dist() (qui initialise alors un groupe
    torch.distributed d'un seul processus) et pour
    image_datasets.load_data() (pas de sharding du dataset).

Le paquet Python mpi4py peut etre installe (`pip install mpi4py`) mais
rester inutilisable si aucun runtime MPI systeme (ex: MS-MPI sous Windows,
OpenMPI/MPICH sous Linux) n'est present : mpi4py leve alors un RuntimeError
("cannot load MPI library") plutot qu'un simple ImportError. On rattrape
donc les deux cas ci-dessous.
"""

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):

    class _FakeComm:
        rank = 0
        size = 1

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, value, root=0):
            return value

        def barrier(self):
            pass

    class _FakeMPI:
        COMM_WORLD = _FakeComm()

    MPI = _FakeMPI()
