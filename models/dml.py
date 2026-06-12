import torch


class DifferentiableMemoryLifecycle:
    """DML: utility update, creation, consolidation, and forgetting.

    This version follows the paper more directly:
        Delta L_t = L_base - L_pred
        u_k <- lambda u_k + (1-lambda) alpha_k Delta L_t
    Utility is then mapped to [0, 1] only for stable thresholding.  Creation,
    consolidation and forgetting follow the lifecycle described in the paper.
    """
    def __init__(self, utility_decay=0.99, theta_new=0.3, theta_low=0.2,
                 core_ratio=0.2, idle_epochs=10, lifecycle_interval=5,
                 core_grad_scale=0.1, utility_center=0.5):
        self.utility_decay = float(utility_decay)
        self.theta_new = float(theta_new)
        self.theta_low = float(theta_low)
        self.core_ratio = float(core_ratio)
        self.idle_epochs = int(idle_epochs)
        self.lifecycle_interval = int(lifecycle_interval)
        self.core_grad_scale = float(core_grad_scale)
        self.utility_center = float(utility_center)

    @torch.no_grad()
    def update_utilities(self, memory, alpha_full, delta_loss):
        if alpha_full is None:
            return
        if not torch.is_tensor(delta_loss):
            delta_loss = torch.tensor(delta_loss, device=memory.utility.device, dtype=memory.utility.dtype)
        # Paper contribution signal: Delta L_t = L_base - L_pred.
        # In practice alpha_k is about 1/K (or 1/top-k); using alpha_k directly
        # keeps all utilities near 0.5 and prevents DML from ever consolidating or
        # forgetting.  We therefore use alpha relative to the active mean, while
        # preserving the sign of Delta L_t.
        d = torch.tanh(delta_loss.detach().to(memory.utility.device, memory.utility.dtype).clamp(-5.0, 5.0))
        alpha = alpha_full.detach().mean(dim=0).to(memory.utility.device, memory.utility.dtype)
        active = memory.active_mask
        active_mean = 1.0 / max(1, int(active.sum().item()))
        rel_alpha = (alpha / active_mean).clamp(0.0, 5.0)
        target = (self.utility_center + 0.25 * rel_alpha * d).clamp(0.0, 1.0)
        memory.utility[active] = self.utility_decay * memory.utility[active] + (1.0 - self.utility_decay) * target[active]

    @torch.no_grad()
    def create_new_memories(self, memory, h_t, alpha_full, semantic_vector=None,
                            tag="new unseen traffic pattern", description="Created online for an unseen traffic pattern.",
                            epoch: int = 0, max_create_per_batch=2, forecast_residuals=None):
        if alpha_full is None:
            return 0
        max_alpha, _ = alpha_full.max(dim=-1)
        unmatched = torch.where(max_alpha < self.theta_new)[0]
        if len(unmatched) == 0:
            return 0
        created = 0
        for row in unmatched[:max_create_per_batch]:
            proto = h_t[row].detach()
            fr = None
            if forecast_residuals is not None:
                fr = forecast_residuals[row].detach()
            ok = memory.add_memory(proto, semantic=semantic_vector, tag=tag, description=description, last_access_epoch=epoch, forecast_residual=fr)
            created += int(ok)
        return created

    @torch.no_grad()
    def consolidate_core(self, memory):
        idx = memory.active_indices
        k = len(idx)
        if k == 0:
            return
        n_core = max(1, int(k * self.core_ratio))
        access = memory.access_count[idx]
        if torch.max(access) > 0:
            access = access / torch.max(access).clamp_min(1e-6)
        vals = memory.utility[idx] + 0.10 * access
        top_pos = torch.topk(vals, k=n_core).indices
        memory.core_mask.zero_()
        memory.core_mask[idx[top_pos]] = True

    @torch.no_grad()
    def forget_obsolete(self, memory, epoch: int):
        idx = memory.active_indices
        if len(idx) == 0:
            return 0
        # The paper removes old low-utility non-core memories; do not artificially
        # protect all initial prototypes.  Keep only a small floor to avoid an empty bank.
        min_keep = max(8, int(0.5 * memory.init_memory_size))
        if len(idx) <= min_keep:
            return 0
        idle = (epoch - memory.last_access[idx]) > self.idle_epochs
        low = memory.utility[idx] < self.theta_low
        noncore = ~memory.core_mask[idx]
        remove = idx[idle & low & noncore]
        if len(idx) - len(remove) < min_keep:
            # Remove the lowest-utility candidates only up to the floor.
            n_remove = max(0, len(idx) - min_keep)
            if n_remove == 0 or len(remove) == 0:
                return 0
            vals = memory.utility[remove]
            remove = remove[torch.topk(-vals, k=min(n_remove, len(remove))).indices]
        memory.deactivate(remove)
        return int(len(remove))

    def before_optimizer_step(self, memory):
        memory.scale_core_gradients(self.core_grad_scale)

    def epoch_end(self, memory, epoch: int):
        self.consolidate_core(memory)
        removed = 0
        if epoch > 0 and epoch % self.lifecycle_interval == 0:
            removed = self.forget_obsolete(memory, epoch)
        return removed
