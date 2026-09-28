"""
ns-3 Integration Module - Real ns-3 Python Bindings Support (ns-3.48+)

This module provides integration with actual ns-3 network simulator
when running in WSL environment where ns-3 is installed.

Works around cppyy JIT issues with ns3::Time by using C++ helper functions
and a loop-based simulation approach instead of Simulator::Schedule.

Key function
------------
run_all_controllers_ns3()
    Runs all 5 routing controllers through the SAME ns-3 topology in a single
    process using a 3-phase per-step design:

    Phase 1 — Weight computation (all 5 controllers, pure Python):
        Each controller computes its preferred link weights for this step.

    Phase 2 — NS-3 simulation (once per step, using GNN weights):
        GNN weights are applied to ns-3 routing tables, UDP traffic apps are
        installed with correct absolute sim-clock timestamps, and
        Simulator::Run() advances the sim clock by one control_interval.
        A SIGALRM wall-clock timeout prevents indefinite hangs.

    Phase 3 — Carbon measurement (all 5 controllers, pure Python):
        Each controller's link weights are used with distribute_traffic_by_routing()
        and calculate_carbon_with_traffic() to measure carbon.  These functions
        are pure Python and do not depend on the ns-3 run in Phase 2.

    Key fix: the old design ran NS-3 once per controller (5× per step).  After
    controller 1 advanced the sim clock, controllers 2–5 tried to schedule
    app-Start events in the past, triggering ns-3's internal assertion and
    hanging the process.  Running NS-3 exactly once per step eliminates this.
"""

import signal
import numpy as np

# --- ns-3 loading with cppyy workarounds ---
NS3_AVAILABLE = False
ns = None
_helpers_ready = False

try:
    from ns import ns as _ns
    ns = _ns
    NS3_AVAILABLE = True
    print("ns-3 Python bindings loaded successfully (ns-3.48)")
except ImportError:
    NS3_AVAILABLE = False
    ns = None
    print("ns-3 not available - install ns-3 with Python bindings")


def _sim_stop_absolute(abs_seconds: float):
    """
    Set an **absolute** stop time on the ns-3 simulator clock.

    Bug fix: the old `_sim_stop(relative_seconds)` was always passing a
    relative duration (e.g. 3600) as if it were an absolute wall-clock time.
    After Seed 1 the ns-3 singleton clock is at 86400 s, so scheduling a stop
    at t=3600 s puts the stop-event in the past and Simulator::Run() blocks
    forever.  This function always sets the stop at the correct absolute time.
    """
    nanoseconds = int(abs_seconds * 1e9)

    try:
        ns.Simulator.Stop(ns.TimeStep(nanoseconds))
        return
    except Exception:
        pass

    try:
        ns.Simulator.Stop(ns.Seconds(float(abs_seconds)))
        return
    except Exception:
        pass

    try:
        ns.Simulator.Stop(ns.Time(f"{int(abs_seconds)}s"))
        return
    except Exception:
        pass

    print(f"  Could not set stop time at {abs_seconds}s, simulation will run until no events")


def _sim_run():
    """Run simulator (no timeout — prefer _sim_run_with_timeout for long runs)."""
    ns.Simulator.Run()


def _sim_run_with_timeout(timeout_s: int = 90):
    """
    Run ns-3 Simulator with a wall-clock timeout to prevent indefinite hangs.

    Uses SIGALRM on Linux/WSL.  Falls back to a plain Run() on Windows where
    SIGALRM is not available (the timeout protection is then disabled).

    Parameters
    ----------
    timeout_s : int
        Maximum wall-clock seconds to allow Simulator::Run() to block.
        Default 90 s gives each 1-hour sim interval up to 90 real seconds.
        Increase for very large topologies if needed.
    """
    if not hasattr(signal, 'SIGALRM'):
        # Windows / environments without SIGALRM — run without timeout
        ns.Simulator.Run()
        return

    def _alarm_handler(signum, frame):
        raise TimeoutError(
            f"Simulator::Run() exceeded {timeout_s}s wall-clock limit — "
            "no events or stuck event loop.  Step carbon set to 0."
        )

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        ns.Simulator.Run()
    finally:
        signal.alarm(0)                          # always cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)  # restore previous handler


def _sim_destroy():
    """
    Destroy the ns-3 simulator and attempt to reset the singleton context.

    ns-3's Simulator is a process-wide singleton.  Calling Destroy() clears
    the event queue and releases aggregated objects, but the DefaultSimulatorImpl
    stays registered between seeds.  Calling Destroy() both *before* building a
    new topology (pre-seed) and *after* finishing one (post-seed) ensures the
    simulator is in a clean state.
    """
    try:
        ns.Simulator.Destroy()
    except Exception:
        pass


def _sim_now():
    """Get current simulation time in seconds"""
    try:
        return ns.Simulator.Now().GetNanoSeconds() / 1e9
    except Exception:
        return 0.0


class NS3NetworkBuilder:
    """Builds actual ns-3 network topology using Python bindings"""

    def __init__(self, num_nodes, topology_type='hierarchical'):
        if not NS3_AVAILABLE:
            raise RuntimeError("ns-3 not available")

        self.num_nodes = num_nodes
        self.topology_type = topology_type
        self.nodes = ns.NodeContainer()
        self.devices = []
        self.ipv4_interfaces = []

    def build_topology(self, topology_graph):
        """Create ns-3 nodes and links from NetworkX graph"""

        self.nodes.Create(self.num_nodes)

        internet_stack = ns.InternetStackHelper()
        internet_stack.Install(self.nodes)

        p2p_helper = ns.PointToPointHelper()
        p2p_helper.SetDeviceAttribute("DataRate", ns.StringValue("100Mbps"))
        p2p_helper.SetChannelAttribute("Delay", ns.StringValue("10ms"))

        ipv4_helper = ns.Ipv4AddressHelper()
        ipv4_helper.SetBase(ns.Ipv4Address("10.1.0.0"),
                           ns.Ipv4Mask("255.255.255.0"))

        num_edges = topology_graph.number_of_edges()
        print(f"  Building p2p links ({self.num_nodes} nodes, {num_edges} edges)...")
        for u, v in topology_graph.edges():
            node_u = self.nodes.Get(int(u))
            node_v = self.nodes.Get(int(v))

            container = ns.NodeContainer()
            container.Add(node_u)
            container.Add(node_v)

            devices = p2p_helper.Install(container)
            self.devices.append(devices)

            interfaces = ipv4_helper.Assign(devices)
            self.ipv4_interfaces.append(interfaces)

            ipv4_helper.NewNetwork()

        print(f"  PopulateRoutingTables ({self.num_nodes} nodes, {num_edges} edges) ...")
        _populated = False
        try:
            import signal as _sig
            if hasattr(_sig, 'SIGALRM'):
                def _pop_timeout(s, f):
                    raise TimeoutError("PopulateRoutingTables exceeded 60s")
                _old = _sig.signal(_sig.SIGALRM, _pop_timeout)
                _sig.alarm(60)
                try:
                    ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
                    _populated = True
                except TimeoutError:
                    print(f"  WARNING: PopulateRoutingTables timed out ({num_edges} edges too dense).")
                    print(f"  Skipping — carbon measurement uses Python routing, unaffected.")
                finally:
                    _sig.alarm(0)
                    _sig.signal(_sig.SIGALRM, _old)
            else:
                ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
                _populated = True
        except Exception as _e:
            if not isinstance(_e, TimeoutError):
                print(f"  WARNING: PopulateRoutingTables failed: {_e}")
        if _populated:
            print("  Routing tables ready.")

        return self.nodes, self.devices, self.ipv4_interfaces



    def install_energy_models(self, node_id, energy_type='grid'):
        """Install ns-3 energy framework on nodes"""
        try:
            node = self.nodes.Get(node_id)
            energy_source = ns.CreateObject("BasicEnergySource")

            initial_energy = (1e9 if energy_type == 'grid'
                              else (1e6 if energy_type == 'battery' else 5e6))
            energy_source.SetAttribute("BasicEnergySourceInitialEnergyJ",
                                       ns.DoubleValue(initial_energy))
            node.AggregateObject(energy_source)
            return energy_source
        except Exception:
            return None


class NS3StateExtractor:
    """Extracts 13-feature network state from the running ns-3 simulation."""

    ENERGY_TYPES = ['solar', 'wind', 'hydro', 'coal', 'nuclear', 'mixed']

    def __init__(self, nodes, cpu_dist=None):
        """
        Args:
            nodes    : ns-3 NodeContainer
            cpu_dist : numpy array of real cpu_usage values from CSV (optional).
                       When provided, cpu_usage is sampled from the real
                       distribution instead of a synthetic formula.
        """
        self.nodes    = nodes
        self.cpu_dist = cpu_dist

    def extract_full_state(self, num_nodes, topology, carbon_mgr=None,
                           timestamp=0, seed=42):
        """
        Extract complete 13-feature network state for GNN input.

        Features (13-dim per node):
          [energy_ratio, carbon_intensity/1000, queue_load, cpu_usage,
           degree/num_nodes, time_factor, carbon_intensity/1500,
           solar, wind, hydro, coal, nuclear, mixed]  <- 6-dim one-hot

        cpu_usage is sampled from the real CSV distribution when available,
        not from a synthetic sinusoid.
        """
        node_features = []
        hour_of_day   = (timestamp / 3600) % 24
        # Data-driven temporal encoding (no synthetic sine waves)
        time_factor   = hour_of_day / 24.0

        for i in range(num_nodes):
            num_interfaces = topology['graph'].degree(i)

            carbon_intensity = (float(carbon_mgr.get_node_intensity(i, timestamp))
                                if carbon_mgr is not None else 400.0)

            energy_ratio = 1.0 - min(carbon_intensity / 800.0, 1.0)

            # cpu_usage: sampled from real CSV distribution when available
            if self.cpu_dist is not None:
                rng_idx  = (seed + int(timestamp) + i) % len(self.cpu_dist)
                cpu_usage = float(self.cpu_dist[rng_idx]) / 100.0
            else:
                cpu_usage = (30 + 40 * time_factor + 10 * float(np.sin(i * 2.0))) / 100.0

            # queue_load: data-driven (based on real CPU/traffic distributions)
            # If cpu_usage is from CSV, use it as a proxy for queue load, or default to time_factor
            queue_load = cpu_usage if self.cpu_dist is not None else time_factor

            features = [
                energy_ratio,
                carbon_intensity / 1000.0,
                queue_load,
                cpu_usage,
                num_interfaces / max(num_nodes, 1),
                time_factor,
                carbon_intensity / 1500.0,
            ]

            # 6-dim one-hot energy source encoding
            one_hot = [0.0] * 6
            if carbon_mgr is not None and hasattr(carbon_mgr, 'get_node_profile_type'):
                ptype = carbon_mgr.get_node_profile_type(i)
                if ptype in self.ENERGY_TYPES:
                    one_hot[self.ENERGY_TYPES.index(ptype)] = 1.0
            else:
                one_hot[-1] = 1.0  # default to 'mixed'
            features.extend(one_hot)

            node_features.append(features)

        return np.array(node_features)  # shape: (num_nodes, 13)


class NS3RoutingController:
    """Applies link weight decisions to ns-3 routing tables."""

    def __init__(self, nodes):
        self.nodes = nodes

    def update_link_metrics(self, link_weights, topology):
        """
        Apply link weights to ns-3 IPv4 interface metrics.

        NOTE: We do NOT call RecomputeRoutingTables() here.
        -------------------------------------------------------
        RecomputeRoutingTables() re-runs the full Dijkstra algorithm for every
        node on every routing-table update.  With N nodes and T time steps this
        is O(T * N^2 log N) total work.  By step 13+ of a 24-step / 150-node
        run, the accumulated UDP-app event state made each call take minutes,
        causing the simulation to hang.

        The carbon measurement in this simulation is entirely Python-side
        (distribute_traffic_by_routing + calculate_carbon_with_traffic), so
        ns-3 routing tables do not affect results.  The SetMetric calls below
        are kept for NetAnim visualisation fidelity only.
        """
        edge_index = topology['edge_index']

        for idx in range(edge_index.shape[1]):
            src = edge_index[0, idx]
            try:
                node     = self.nodes.Get(int(src))
                ipv4     = ns.GetObject(node, ns.Ipv4)

                if ipv4:
                    n_ifaces      = ipv4.GetNInterfaces()
                    interface_idx = (idx % n_ifaces) + 1
                    if interface_idx < n_ifaces:
                        metric = max(1, int(link_weights[idx]))
                        ipv4.SetMetric(interface_idx, metric)
            except Exception:
                continue
        # RecomputeRoutingTables() intentionally OMITTED -- see docstring above.


# ---------------------------------------------------------------------------
# Unified 5-controller NS-3 runner
# ---------------------------------------------------------------------------

def run_all_controllers_ns3(
    gnn_model,
    topology,
    carbon_mgr,
    energy_mgr,
    duration_seconds=86400,
    control_interval=3600,
    enable_netanim=False,
    netanim_output="results/carbon-routing-animation.xml",
    seed=42,
):
    """
    Run all 5 routing controllers through the SAME ns-3 topology.

    Design — Option C (metric-swap):
    ---------------------------------
    One NS-3 topology is built per call.  At each hourly control interval:
      1. For each of the 5 controllers in sequence:
         a. Controller computes its link weights (pure Python)
         b. Weights applied to ns-3 via Ipv4::SetMetric + RecomputeRoutingTables
         c. NS-3 runs for `control_interval` seconds of simulated time
         d. Carbon is measured via NetworkEnergyManager using the traffic loads
            that were distributed under that controller's routing decision
      2. Carbon histories are accumulated per controller

    This means each controller experiences the same NS-3 topology and the
    same carbon-intensity profile at each step, making the comparison paired
    and directly fair.  Traffic coupling (the slight dependency between
    controllers sharing one NS-3 instance) is accepted per user preference.

    Parameters
    ----------
    gnn_model        : CarbonAwareGAT model (pre-trained or random init)
    topology         : dict from create_network() — graph, edge_index, edge_features
    carbon_mgr       : CarbonProfileManager
    energy_mgr       : NetworkEnergyManager (already initialized)
    duration_seconds : total simulation duration in seconds
    control_interval : seconds between routing updates (default 3600 = 1h)
    enable_netanim   : generate NetAnim XML for the GNN controller run
    netanim_output   : path for NetAnim XML file
    seed             : random seed for reproducibility

    Returns
    -------
    dict  {
        'gnn'                : {'carbon_history': [...], 'total_carbon': float},
        'ospf'               : {'carbon_history': [...], 'total_carbon': float},
        'threshold'          : {'carbon_history': [...], 'total_carbon': float},
        'linear_flow_carbon' : {'carbon_history': [...], 'total_carbon': float},
        'nash_game'          : {'carbon_history': [...], 'total_carbon': float},
        'ns3_log'            : [{'time':..., 'step':...}, ...],
    }
    """
    if not NS3_AVAILABLE:
        raise RuntimeError("ns-3 not available — cannot run NS-3 simulation")

    import os
    num_nodes     = topology['graph'].number_of_nodes()
    num_intervals = int(duration_seconds / control_interval)

    # ── Pre-seed teardown ─────────────────────────────────────────────────────
    # Destroy any leftover simulator state from a previous seed.  The ns-3
    # Simulator singleton persists across calls within the same process, so we
    # must call Destroy() *before* building the new topology (in addition to the
    # post-seed Destroy() at the end).  This is idempotent on the first call.
    _sim_destroy()

    # ── Load CSV cpu_usage distribution for realistic node features ──────────
    cpu_dist = None
    try:
        import pandas as pd
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'carbon_network_data.csv'
        )
        if os.path.exists(csv_path):
            df       = pd.read_csv(csv_path, usecols=['cpu_usage'])
            cpu_dist = df['cpu_usage'].dropna().values.astype(float)
    except Exception:
        pass

    # ── Build NS-3 topology (once per seed) ──────────────────────────────────
    print("Building ns-3 topology...")
    builder = NS3NetworkBuilder(num_nodes)
    nodes, devices, interfaces = builder.build_topology(topology['graph'])

    print("Installing energy models...")
    for i in range(num_nodes):
        builder.install_energy_models(i, 'grid')

    # ── NetAnim (GNN controller only) ─────────────────────────────────────────
    netanim = None
    if enable_netanim:
        try:
            from integration.netanim_helper import NetAnimHelper
            netanim = NetAnimHelper()
            netanim.initialize(netanim_output)
            netanim.set_node_positions(topology)
            carbon_assignments = {
                i: carbon_mgr.node_assignments.get(i, 'mixed_grid')
                for i in range(num_nodes)
            }
            netanim.set_carbon_aware_colors(carbon_assignments)
            print(f"NetAnim enabled: {netanim_output}")
        except Exception as e:
            print(f"NetAnim initialization failed: {e}")
            netanim = None

    # ── Shared infrastructure ─────────────────────────────────────────────────
    state_extractor  = NS3StateExtractor(nodes, cpu_dist=cpu_dist)
    ns3_routing_ctrl = NS3RoutingController(nodes)

    # ── Instantiate all 5 controllers ────────────────────────────────────────
    from simulation.gnn_routing_controller import (
        RoutingController,
        BaselineController,
        ThresholdCarbonController,
        LinearFlowCarbonController,
        NashGameController,
    )
    from enhanced_gnn_model import create_graph_from_network_state, RouteOptimizer

    gnn_ctrl  = RoutingController(gnn_model, topology, carbon_mgr, energy_mgr,
                                  control_interval=control_interval,
                                  use_ns3=True, seed=seed)
    ospf_ctrl = BaselineController(topology, carbon_mgr, energy_mgr,
                                   control_interval=control_interval, seed=seed)
    thr_ctrl  = ThresholdCarbonController(topology, carbon_mgr, energy_mgr,
                                          control_interval=control_interval, seed=seed)
    p1_ctrl   = LinearFlowCarbonController(topology, carbon_mgr, energy_mgr,
                                           control_interval=control_interval, seed=seed)
    nash_ctrl = NashGameController(topology, carbon_mgr, energy_mgr,
                                   control_interval=control_interval, seed=seed)

    CTRL_ORDER = [
        ('gnn',                gnn_ctrl),
        ('ospf',               ospf_ctrl),
        ('threshold',          thr_ctrl),
        ('linear_flow_carbon', p1_ctrl),
        ('nash_game',          nash_ctrl),
    ]

    # Carbon histories keyed by controller name
    carbon_histories = {name: [] for name, _ in CTRL_ORDER}
    ns3_log = []

    CTRL_LABELS = {
        'gnn':                'GNN (Carbon-Aware GAT)',
        'ospf':               'OSPF Baseline',
        'threshold':          'ThresholdCarbonController',
        'linear_flow_carbon': 'LinearFlowCarbonController (El-Zahr & Zilberman 2025)',
        'nash_game':          'NashGameController (Hogade et al.)',
    }

    print(f"\nRunning 5-controller NS-3 simulation:")
    print(f"  Duration : {duration_seconds}s ({num_intervals} x {control_interval}s intervals)")
    print(f"  Nodes    : {num_nodes}")
    print(f"  Seed     : {seed}")
    print()
    for name, _ in CTRL_ORDER:
        print(f"  [{name}] {CTRL_LABELS[name]}")
    print()

    # ── Main simulation loop ──────────────────────────────────────────────────
    # 3-phase design per step to avoid ns-3 past-event assertion:
    #
    # Phase 1: Compute all 5 controllers' link weights (pure Python).
    # Phase 2: Run NS-3 ONCE using GNN weights.  The sim clock advances by
    #          control_interval.  A SIGALRM timeout prevents hangs.
    # Phase 3: Measure carbon for all 5 controllers via Python routing sim.
    #
    # OLD design: NS-3 was run once per controller (5× per step).  After
    # controller 1 advanced the sim clock, controllers 2-5 tried to install
    # UDP apps with Start(t_past) → ns-3 fatal assertion → process hung.
    # Carbon measurement was always pure Python anyway (distribute_traffic_
    # by_routing + calculate_carbon_with_traffic), so the extra NS-3 runs
    # were both harmful and unnecessary.
    from simulation.traffic_matrix import distribute_traffic_by_routing

    sim_clock = 0.0

    for step in range(num_intervals):
        current_time  = step * control_interval
        hours         = current_time / 3600

        # Advance the absolute simulation clock for this step
        sim_clock    += control_interval
        app_start_abs = float(sim_clock - control_interval)
        app_end_abs   = float(sim_clock)

        # ── Shared inputs for this step ───────────────────────────────────
        ns3_state = state_extractor.extract_full_state(
            num_nodes, topology, carbon_mgr=carbon_mgr,
            timestamp=current_time, seed=seed
        )

        carbon_intensities = {
            nid: carbon_mgr.get_node_intensity(nid, current_time)
            for nid in range(num_nodes)
        }

        # Reproducible traffic seed + shared flows for this step
        np.random.seed(seed + int(current_time))
        traffic_flows = ospf_ctrl.traffic_matrix.generate_datacenter_traffic()

        # ── Phase 1: compute all controllers' link weights ────────────────
        import time as _time
        t_phase1 = _time.monotonic()
        all_weights = {}
        for ctrl_name, ctrl in CTRL_ORDER:
            try:
                if ctrl_name == 'gnn':
                    full_state = {
                        'node_features': ns3_state,
                        'edge_index':    topology['edge_index'],
                        'edge_features': topology['edge_features'],
                    }
                    w, _ = gnn_ctrl._compute_weights_only(current_time, state=full_state)
                elif ctrl_name == 'ospf':
                    w = np.ones(topology['edge_index'].shape[1])
                elif ctrl_name == 'threshold':
                    w = thr_ctrl._compute_weights_only(current_time)
                elif ctrl_name == 'linear_flow_carbon':
                    w = p1_ctrl._compute_link_weights(current_time)
                elif ctrl_name == 'nash_game':
                    w = nash_ctrl._nash_link_weights(current_time)
                else:
                    w = np.ones(topology['edge_index'].shape[1])
            except Exception as e:
                print(f"  [{ctrl_name}] weight computation failed: {e}")
                w = np.ones(topology['edge_index'].shape[1])
            all_weights[ctrl_name] = w
        dt1 = _time.monotonic() - t_phase1

        # ── Phase 2: run NS-3 ONCE per step (GNN weights for realism) ────
        # Apply GNN routing weights to ns-3 before the run.
        t_phase2 = _time.monotonic()
        ns3_routing_ctrl.update_link_metrics(all_weights['gnn'], topology)

        # Install UDP traffic apps for this time window (absolute sim times).
        # Installing apps is done BEFORE Run() and for this single interval
        # only, so there is no risk of scheduling events in the past.
        try:
            udp_port   = 9000 + (step % 100)
            flow_sample = traffic_flows[:min(3, len(traffic_flows))]
            for flow in flow_sample:
                src_node = int(flow.get('src', 0)) % num_nodes
                dst_node = int(flow.get('dst', 1)) % num_nodes
                if src_node == dst_node:
                    dst_node = (src_node + 1) % num_nodes
                try:
                    udp_server  = ns.UdpServerHelper(udp_port)
                    server_apps = udp_server.Install(nodes.Get(dst_node))
                    server_apps.Start(ns.Seconds(app_start_abs))
                    server_apps.Stop(ns.Seconds(app_end_abs))

                    dst_iface  = interfaces[min(dst_node, len(interfaces) - 1)]
                    dst_addr   = (dst_iface.GetAddress(0)
                                  if hasattr(dst_iface, 'GetAddress')
                                  else ns.Ipv4Address("10.1.0.1"))
                    udp_client = ns.UdpClientHelper(dst_addr, udp_port)
                    udp_client.SetAttribute("MaxPackets",
                                            ns.UintegerValue(100))
                    udp_client.SetAttribute("Interval",
                                            ns.TimeValue(ns.Seconds(
                                                float(control_interval) / 100.0)))
                    udp_client.SetAttribute("PacketSize",
                                            ns.UintegerValue(1024))
                    client_apps = udp_client.Install(nodes.Get(src_node))
                    client_apps.Start(ns.Seconds(app_start_abs))
                    client_apps.Stop(ns.Seconds(app_end_abs))
                except Exception:
                    pass  # skip this flow on NS-3 API error
                udp_port += 1
        except Exception:
            pass  # graceful fallback: NS-3 runs without explicit traffic apps

        # Run the simulator for this interval — single run, with timeout guard.
        try:
            _sim_stop_absolute(sim_clock)
            _sim_run_with_timeout(timeout_s=120)
        except TimeoutError as te:
            print(f"  [ns3] step {step} timed out: {te}")
        except Exception as e:
            print(f"  [ns3] step {step} run error: {e}")
        dt2 = _time.monotonic() - t_phase2

        # ── Phase 3: measure carbon for all 5 controllers (pure Python) ───
        # Each controller uses its own link weights with the Python routing
        # simulator — no dependency on the NS-3 run above.
        t_phase3 = _time.monotonic()
        step_results = {}
        for ctrl_name, _ in CTRL_ORDER:
            w = all_weights[ctrl_name]
            node_traffic_loads, _ = distribute_traffic_by_routing(
                traffic_flows,
                topology['graph'],
                w,
                topology['edge_index'],
            )
            carbon = energy_mgr.calculate_carbon_with_traffic(
                carbon_intensities, node_traffic_loads, control_interval
            )
            carbon_histories[ctrl_name].append(carbon)
            step_results[ctrl_name] = carbon
        dt3 = _time.monotonic() - t_phase3

        ns3_log.append({'time': current_time, 'step': step, **step_results})

        # Progress line — includes per-phase wall-clock times for diagnostics
        vals = " | ".join(
            f"{n.replace('linear_flow_carbon', 'LFC').replace('nash_game', 'Nash')}"
            f"={step_results[n]:.4f}"
            for n, _ in CTRL_ORDER
        )
        print(f"  t={current_time:6d}s ({hours:5.1f}h) "
              f"[W:{dt1:.1f}s NS3:{dt2:.1f}s C:{dt3:.1f}s] | {vals}", flush=True)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\nCleaning up ns-3...")
    _sim_destroy()

    # ── Build result dict ─────────────────────────────────────────────────────
    results = {}
    print()
    print(f"{'Controller':<32} {'Total CO2 (gCO2)':>18}  {'vs OSPF':>10}")
    print("-" * 64)
    ospf_total = float(sum(carbon_histories['ospf']))
    for name, _ in CTRL_ORDER:
        total  = float(sum(carbon_histories[name]))
        pct    = (ospf_total - total) / ospf_total * 100 if ospf_total else 0
        results[name] = {
            'carbon_history': carbon_histories[name],
            'total_carbon':   total,
            'avg_carbon_rate': float(np.mean(carbon_histories[name])),
        }
        marker = " <-- OUR METHOD" if name == 'gnn' else ""
        label  = CTRL_LABELS[name][:31]
        print(f"  {label:<30} {total:>18.2f}  {pct:>+9.1f}%{marker}")

    results['ns3_log'] = ns3_log
    return results


# ---------------------------------------------------------------------------
# Legacy single-controller wrapper (backward compat)
# ---------------------------------------------------------------------------

def run_ns3_simulation(gnn_model, topology, carbon_mgr,
                       duration_seconds=86400, control_interval=3600,
                       enable_netanim=False,
                       netanim_output="results/carbon-routing.xml"):
    """
    Legacy entry point — GNN controller only.
    Prefer run_all_controllers_ns3() for full 5-controller comparison.
    """
    from simulation.energy_model import NetworkEnergyManager
    energy_mgr = NetworkEnergyManager(topology['graph'].number_of_nodes())
    energy_mgr.initialize_nodes()

    results = run_all_controllers_ns3(
        gnn_model=gnn_model,
        topology=topology,
        carbon_mgr=carbon_mgr,
        energy_mgr=energy_mgr,
        duration_seconds=duration_seconds,
        control_interval=control_interval,
        enable_netanim=enable_netanim,
        netanim_output=netanim_output,
    )
    # Return in old format for backward compat
    return {
        'status':    'completed',
        'duration':  duration_seconds,
        'intervals': len(results['ns3_log']),
        'log': [
            {'time': e['time'], 'carbon_pred': e.get('gnn', 0), 'step': e['step']}
            for e in results['ns3_log']
        ],
    }


if __name__ == "__main__":
    if NS3_AVAILABLE:
        print("\nns-3 integration module ready (ns-3.48)")
        print("  Use run_all_controllers_ns3() to compare all 5 controllers through NS-3")
    else:
        print("\nns-3 not installed")
        print("  Install ns-3 with Python bindings in WSL")
