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

        num_g_per_class = global_prototypes.shape[0]//self.num_classes # g
        global_prototypes = global_prototypes.reshape(num_g_per_class, self.num_classes, global_prototypes.shape[1])  # shape (g, c, D)

        num_p_per_class = part_prototypes.shape[0]//self.num_classes  # p
        part_prototypes = part_prototypes.reshape(num_p_per_class, self.num_classes, part_prototypes.shape[1]) # shape (p, c, D)

        # we need to entail each part prototype with at least one gloabl prototype of its own class
        # so we repeat_interleave the dim 0 of the part_prototypes and repeat the dim 0 of the global prototype
        # so the result is  (g0,g1,...,g9,g0,g1,...,g9,g0,...,g9...)
        # a              nd (p0,p0,...,p0,p1,p1,...,p1,p2,...,p2...)
        # so the angles are (a0,a0,...,a0,a1,a1,...,a1,a2,...,a2...)
        # and we reshape to (p, g, c)
        #                   [(a0,a0,...,a0),
        #                    (a1,a1,...,a1), 
        #                    (a2,a2,...,a2)]
        # and then we get the minimum of each row (min in dim 1) to get the entailment loss of each local prototype
        # and then we sum them up to get the total entailment loss of each class, and then sum or mean them across classes

        global_prototypes = global_prototypes.repeat(num_p_per_class, 1, 1)  # shape (p*g, c, D)
        part_prototypes = part_prototypes.repeat_interleave(num_g_per_class, dim=0)  # shape (p*g, c, D)

        _angle = L.oxy_angle(global_prototypes, part_prototypes, _curv)  # shape (P)
        _aperture = L.half_aperture(global_prototypes, _curv)  # shape (P)
        entailment_loss = torch.clamp(_angle - _aperture, min=0)  # shape (P)

        # reshape to (p, g, c)
        entailment_loss = entailment_loss.reshape(num_p_per_class, num_g_per_class, self.num_classes)
        # get the minimum of each row (dim 1) = entailment loss of each local prototype
        entailment_loss = entailment_loss.min(dim=1).values  # shape (p, c)
        # sum them up to get the total entailment loss of each class
        entailment_loss = entailment_loss.sum(dim=0)
        # sum or mean them across classes
        if self.reduction == "mean":
            entailment_loss = entailment_loss.mean()
        elif self.reduction == "sum":
            entailment_loss = entailment_loss.sum()
        return self.loss_weight * entailment_loss
