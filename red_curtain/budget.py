"""Pure VRAM budget calculation; no GPU-size-specific hard caps."""
from dataclasses import dataclass, asdict
import math

GIB = 1024**3
WORKSPACE_RESERVE = int(1.5*GIB)
MIN_CACHE = 64*1024**2


def allocator_ceiling(total_bytes,free_bytes,reserved_bytes=0):
    if total_bytes<=0 or not 0<=free_bytes<=total_bytes or not 0<=reserved_bytes<=total_bytes:
        raise ValueError('Invalid GPU memory accounting')
    external_reserve=max(512*1024**2,int(total_bytes*.05))
    # Reserved memory already belongs to this process; it is not a competing allocation.
    return max(0,min(int(total_bytes*.9),free_bytes+reserved_bytes-external_reserve))


@dataclass(frozen=True)
class MemoryBudget:
    limit_bytes:int
    cache_bytes:int
    fixed_bytes:int
    workspace_reserve_bytes:int
    mode:str

    def to_dict(self):
        return asdict(self)


def resolve_budget(total_bytes,free_bytes,reserved_bytes,fixed_bytes,requested_gib=None):
    if fixed_bytes < 0:
        raise ValueError('Invalid fixed-weight memory estimate')
    ceiling=allocator_ceiling(total_bytes,free_bytes,reserved_bytes)
    if requested_gib is not None and (not math.isfinite(requested_gib) or requested_gib<1):
        raise ValueError('VRAM limit must be at least 1 GiB, or automatic')
    limit=ceiling if requested_gib is None or requested_gib>=ceiling/GIB else int(requested_gib*GIB)
    minimum=fixed_bytes+WORKSPACE_RESERVE+MIN_CACHE
    if limit<minimum:
        raise MemoryError(f'This checkpoint needs at least {minimum/GIB:.2f} GiB of allocator budget '
                          f'(compute weights, workspace reserve and minimum expert cache); '
                          f'only {limit/GIB:.2f} GiB is available under the current limit and GPU usage. '
                          'Free GPU memory or use a GPU with more VRAM. System-RAM offloading remains enabled.')
    return MemoryBudget(limit,limit-fixed_bytes-WORKSPACE_RESERVE,fixed_bytes,
                        WORKSPACE_RESERVE,'automatic' if requested_gib is None else 'manual ceiling')
