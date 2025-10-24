import torch

class GradCAM:
    """
    Simple realization of Grad-CAM.
    target_layer — convolutional layer/block from which we read activations and gradients.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.inference_mode(False)
    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: [1,3,H,W] (normalized)
        return: (cam[H',W'] в [0,1], class_idx)
        """
        self.model.zero_grad(set_to_none=True)
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        score = out[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients   # [N, C, H, W]
        acts  = self.activations # [N, C, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).relu()  # [N, H, W]
        cam = cam[0].cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam, class_idx
