# Lightweight UAV–DTN simulation skeleton
# ----------------------------------------
# This script creates:
#   • Ground nodes placed uniformly at random in a square
#   • A message generator (Poisson arrivals, fixed deadlines)
#   • A simple “Single‑Route” policy:
#       – Build a TSP tour with a greedy nearest‑neighbour heuristic
#       – N_uav UAVs are evenly spaced on that tour
#   • Time‑stepped simulation loop (store‑carry‑forward)
#   • End‑of‑run statistics: on‑time delivery ratio, average delay
#
# You can extend/replace the Policy class to plug in MRT‑Grid, HoP‑DTN
# or your future DRL agent.
#
# Save this file (or copy into a Jupyter notebook cell) and run.
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random, itertools, math
from collections import deque, namedtuple
import time, uuid

Message = namedtuple("Message", "id src dst gen_t deadline delivered_t")

class GroundNode:
    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = np.array(pos, dtype=float)
        self.queue = deque()          # Messages waiting for pickup

class UAV:
    def __init__(self, idx, pos, speed, capacity):
        self.idx = idx
        self.pos = np.array(pos, dtype=float)
        self.speed = speed
        self.capacity = capacity
        self.buffer = []              # onboard msgs
        self.waypoints = []           # sequence of node indices
        self.wp_idx = 0               # current waypoint pointer

class SingleRoutePolicy:
    """Pre‑computes a TSP tour (greedy) over all ground nodes."""
    def __init__(self, nodes, n_uav):
        self.nodes = nodes
        self.tour = self._greedy_tsp([n.pos for n in nodes])
        self.n_uav = n_uav
    
    @staticmethod
    def _greedy_tsp(points):
        n = len(points)
        remaining = set(range(1, n))
        tour = [0]
        while remaining:
            last = tour[-1]
            nxt = min(remaining, key=lambda j: np.linalg.norm(points[last]-points[j]))
            tour.append(nxt)
            remaining.remove(nxt)
        return tour
    
    def initial_uav_states(self, speed, capacity):
        # Evenly space Uavs along tour
        positions = [self.nodes[i].pos for i in self.tour]
        cumdist = np.cumsum([0]+[np.linalg.norm(positions[i]-positions[i-1]) for i in range(1,len(positions))])
        total = cumdist[-1]
        spacing = total/self.n_uav
        uavs=[]
        for k in range(self.n_uav):
            target_dist = k*spacing
            idx = np.searchsorted(cumdist, target_dist)-1
            segfrac = (target_dist-cumdist[idx])/(cumdist[idx+1]-cumdist[idx]+1e-9)
            start_pos = positions[idx]*(1-segfrac)+positions[idx+1]*segfrac
            u = UAV(k, start_pos, speed, capacity)
            u.waypoints = self.tour
            uavs.append(u)
        return uavs
    
    def next_target(self, uav):
        uav.wp_idx = (uav.wp_idx+1)%len(uav.waypoints)
        return self.nodes[uav.waypoints[uav.wp_idx]].pos

class Simulator:
    def __init__(self, n_nodes=25, area=100, n_uav=5, speed=1.0, capacity=30,
                 lam=1.0, deadline=100, comm_range=5, sim_time=1000, seed=1):
        random.seed(seed); np.random.seed(seed)
        self.time=0
        self.dt=1
        self.comm_range=comm_range
        self.deadline=deadline
        self.sim_time=sim_time
        # Create ground nodes
        self.nodes=[GroundNode(i,np.random.rand(2)*area) for i in range(n_nodes)]
        # Policy
        self.policy=SingleRoutePolicy(self.nodes, n_uav)
        self.uavs=self.policy.initial_uav_states(speed,capacity)
        # Traffic
        self.lam=lam
        self.next_arrival=self._exp_sample()
        self.msgs_generated=0
        self.delivered=[]
        self.dropped=0
    
    def _exp_sample(self):
        return self.time+np.random.exponential(1/self.lam)
    
    def _generate_message(self):
        src,dst=random.sample(self.nodes,2)
        m=Message(str(uuid.uuid4()),src.idx,dst.idx,self.time,self.time+self.deadline,None)
        src.queue.append(m)
        self.msgs_generated+=1
    
    def _within_range(self, pos1,pos2):
        return np.linalg.norm(pos1-pos2)<=self.comm_range
    
    def run(self, verbose=False):
        tic=time.time()
        while self.time<self.sim_time:
            # traffic arrivals
            if self.time>=self.next_arrival:
                self._generate_message()
                self.next_arrival=self._exp_sample()
            # UAV actions
            for u in self.uavs:
                # move toward current target
                if not u.waypoints: continue
                target=self.nodes[u.waypoints[u.wp_idx]].pos
                vec=target-u.pos
                dist=np.linalg.norm(vec)
                step=min(u.speed*self.dt, dist)
                if dist>1e-6:
                    u.pos += vec/dist*step
                if dist<=1e-3:  # reached
                    # exchange with node
                    node=self.nodes[u.waypoints[u.wp_idx]]
                    # deliver
                    undel=[]
                    for m in u.buffer:
                        if m.dst==node.idx:
                            self.delivered.append(m._replace(delivered_t=self.time))
                        else:
                            undel.append(m)
                    u.buffer=undel
                    # pick up
                    while node.queue and len(u.buffer)<u.capacity:
                        u.buffer.append(node.queue.popleft())
                    # advance waypoint
                    target=self.policy.next_target(u)
                # drop expired msgs
                still=[]
                for m in u.buffer:
                    if self.time>m.deadline:
                        self.dropped+=1
                    else:
                        still.append(m)
                u.buffer=still
            # Messages expiring in queues
            for n in self.nodes:
                keep=deque()
                while n.queue:
                    m=n.queue.popleft()
                    if self.time>m.deadline:
                        self.dropped+=1
                    else:
                        keep.append(m)
                n.queue=keep
            self.time+=self.dt
        toc=time.time()
        if verbose:
            print(f"Sim finished in {toc-tic:.2f}s")
        self.report()
    
    def report(self):
        delivered_on_time=len(self.delivered)
        avg_delay=np.mean([m.delivered_t-m.gen_t for m in self.delivered]) if self.delivered else None
        print(f"Generated: {self.msgs_generated}")
        print(f"Delivered on‑time: {delivered_on_time}  ({delivered_on_time/self.msgs_generated*100:.1f}%)")
        print(f"Dropped/late: {self.dropped}")
        print(f"Average delay (delivered): {avg_delay:.1f}" if avg_delay else "No deliveries")

# --- MINI‑VISUALISATION WRAPPER ------------------------------------------
def visualise_sim(sim, interval=200, save_as=None):
    """
    Animates a finished Simulator *sim* (after sim.run()).
    * interval : milliseconds between frames
    * save_as  : filename.gif or .mp4  (requires imagemagick / ffmpeg)
    """
    # collect node + UAV traces
    node_xy = np.array([n.pos for n in sim.nodes])
    uav_trails = {u.idx: [u.pos.copy()] for u in sim.uavs}
    # rerun but store snapshots
    sim.time = 0
    sim.delivered.clear(); sim.dropped=0
    frames=[]
    while sim.time < sim.sim_time:
        frames.append((sim.time, [u.pos.copy() for u in sim.uavs]))
        sim._generate_message() if sim.time>=sim.next_arrival else None
        for u in sim.uavs:
            target = sim.nodes[u.waypoints[u.wp_idx]].pos
            vec, dist = target-u.pos, np.linalg.norm(target-u.pos)
            step = min(u.speed, dist)
            if dist > 1e-6:
                u.pos += vec/dist*step
            if dist <= 1e-3:
                u.wp_idx=(u.wp_idx+1)%len(u.waypoints)
            uav_trails[u.idx].append(u.pos.copy())
        sim.time += 1
    # plotting
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    lim = np.max(node_xy)+5
    ax.set_xlim(-5, lim); ax.set_ylim(-5, lim)
    ax.set_title("UAV‑DTN Message Ferrying")
    node_scatter = ax.scatter(node_xy[:,0], node_xy[:,1], c='black', s=20, label='Ground nodes')
    uav_scatter  = ax.scatter([], [], c='red',  s=40, label='UAVs')
    ax.legend(loc='upper right')
    def update(frame):
        t, positions = frame
        xy = np.array(positions)
        uav_scatter.set_offsets(xy)
        return uav_scatter,
    ani = FuncAnimation(fig, update, frames=frames,
                        interval=interval, blit=True, repeat=False)
    if save_as:
        ani.save(save_as, dpi=100)
    plt.show()

# --- quick demo -------------------------------------------------------------
if __name__=="__main__":
    sim=Simulator(n_nodes=25, n_uav=5, lam=0.8, deadline=80, sim_time=500)
    sim.run(verbose=True)
    visualise_sim(sim, interval=100)
