import torch
import tools.lorentz as L

class Entailment(object):  # TODO Check
    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        print(f"setup Entailment Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, model):
        if self.loss_weight == 0:
            return torch.tensor(0, device=model.device)

        model.curv.data = torch.clamp(model.curv.data, **model._curv_minmax)
        _curv = model.curv.exp()

        # Option1: for now, we find the entailment of each global_prot with its corresponding part-prots (only positive samples)
        # TODO use pairwise_inner for calculating the inner product of all elements later!
        global_prototypes = model.prototype_vectors_global.squeeze()  # shape (num_classes, D)
        part_prototypes = model.prototype_vectors.squeeze()  # shape (P, D)

        #hyper_global_prototypes = L.get_hyperbolic_feats(global_prototypes, model.visual_alpha, model.curv, model.device)
        #hyper_part_prototypes = L.get_hyperbolic_feats(part_prototypes, model.visual_alpha, model.curv, model.device)

        # positive samples only. need to repeat the global prots to match the size of the part-ones
        #num_prot_per_class = part_prototypes.shape[0]//self.num_classes
        #global_prototypes = global_prototypes.repeat_interleave(num_prot_per_class, dim=0)  # shape (P, D)

        _angle = L.oxy_angle(global_prototypes, part_prototypes, _curv)  # shape (P)
        _aperture = L.half_aperture(global_prototypes, _curv)  # shape (P)
        entailment_loss = torch.clamp(_angle - _aperture, min=0)  # shape (P)
        if self.reduction == "mean":
            entailment_loss = entailment_loss.mean()
        elif self.reduction == "sum":
            entailment_loss = entailment_loss.sum()
        return self.loss_weight * entailment_loss
