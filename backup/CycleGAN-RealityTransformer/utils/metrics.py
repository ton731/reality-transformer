import torch
from skimage.metrics import structural_similarity


def image_similarity(realA, cycleA):
    # cycleA: A --genB--> B --genA--> A
    # SSIM: -1~1, 1 is the best
    ssim = 0
    for i in range(realA.shape[0]):
        real = realA[i].detach().cpu().numpy()
        cycle = cycleA[i].detach().cpu().numpy()
        ssim += structural_similarity(real, cycle,
                                      data_range=cycle.max()-cycle.min(),
                                      channel_axis=0)
    return ssim




class CycleGANMetric:

    def __init__(self, name="train", use_identity=False, num_epoch=300):

        self.name = name
        self.use_identity = use_identity
        self.sample_count = 0

        # discriminator: probability
        self.discriminatorA_predict_realA = [0 for _ in range(num_epoch)]
        self.discriminatorA_predict_fakeA = [0 for _ in range(num_epoch)]
        self.discriminatorB_predict_realB = [0 for _ in range(num_epoch)]
        self.discriminatorB_predict_fakeB = [0 for _ in range(num_epoch)]

        # generator: SSIM
        self.generator_A2B2A = [0 for _ in range(num_epoch)]
        self.generator_B2A2B = [0 for _ in range(num_epoch)]

        # identity: SSIM
        self.identity_A2A = [0 for _ in range(num_epoch)]
        self.identity_B2B = [0 for _ in range(num_epoch)]


    @torch.no_grad()
    def batch_update(self, epoch, 
                     D_A_real, D_A_fake, 
                     D_B_real, D_B_fake, 
                     cycleA, cycleB, 
                     A, B,
                     identityA=None, identityB=None):
        
        batch_size = D_A_real.shape[0]
        self.sample_count += batch_size

        # discriminator: probability
        self.discriminatorA_predict_realA[epoch] += torch.mean(D_A_real, dim=[2, 3]).sum().item()
        self.discriminatorA_predict_fakeA[epoch] += torch.mean(D_A_fake, dim=[2, 3]).sum().item()
        self.discriminatorB_predict_realB[epoch] += torch.mean(D_B_real, dim=[2, 3]).sum().item()
        self.discriminatorB_predict_fakeB[epoch] += torch.mean(D_B_fake, dim=[2, 3]).sum().item()

        # generator: SSIM
        self.generator_A2B2A[epoch] += image_similarity(A, cycleA)
        self.generator_B2A2B[epoch] += image_similarity(B, cycleB)

        # identity: SSIM
        if self.use_identity:
            self.identity_A2A[epoch] += image_similarity(A, identityA)
            self.identity_B2B[epoch] += image_similarity(B, identityB)

    
    def epoch_update(self, epoch):
        
        N = self.sample_count

        # divide the values with sample number
        self.discriminatorA_predict_realA[epoch] /= N
        self.discriminatorA_predict_fakeA[epoch] /= N
        self.discriminatorB_predict_realB[epoch] /= N
        self.discriminatorB_predict_fakeB[epoch] /= N
        self.generator_A2B2A[epoch] /= N
        self.generator_B2A2B[epoch] /= N
        if self.use_identity:
            self.identity_A2A[epoch] /= N
            self.identity_B2B[epoch] /= N

        self.sample_count = 0

    
    def info(self, epoch):
        out = "\n"
        out += f"dataset: {self.name} \n"
        out += f"discA_real: {self.discriminatorA_predict_realA[epoch]:.4f}, discA_fake: {self.discriminatorA_predict_fakeA[epoch]:.4f} \n"
        out += f"discB_real: {self.discriminatorB_predict_realB[epoch]:.4f}, discB_fake: {self.discriminatorB_predict_fakeB[epoch]:.4f} \n"
        out += f"cycleA: {self.generator_A2B2A[epoch]:.4f}, cycleB: {self.generator_B2A2B[epoch]:.4f} \n"
        if self.use_identity:
            out += f"identityA: {self.identity_A2A[epoch]:.4f}, identityB: {self.identity_B2B[epoch]:.4f} \n"
        out += "="*100
        out += "\n"*2
        return out







def test_ssim():
    import torch
    import time
    start = time.time()

    for i in range(1000):
        # img1 = torch.randn(3, 256, 256).numpy()
        # img2 = torch.randn(3, 256, 256).numpy()
        img1 = torch.randn(5, 3, 256, 256)
        img2 = torch.randn(5, 3, 256, 256)
        ssim = image_similarity(img1, img2)
        end = time.time()
        print(f"i: {i}, ssim: {ssim}")

    print("time:", end-start)

    # 5000 images take 53 seconds (use numpy and single image)
    # 5000 images take 54 seconds (use torch and batch size = 1)
    # 5000 images take 53 seconds (use torch and batch size = 5)



if __name__ == "__main__":
    test_ssim()


