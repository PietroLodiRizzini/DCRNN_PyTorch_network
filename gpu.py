
class GPU:
    def __init__(self, gpu_id):
        self._gpu_id = gpu_id

    def get_gpu_id(self):
        return self._gpu_id

    def set_gpu_id(self, new_id):
        self._gpu_id = new_id

gpu = GPU(1)
