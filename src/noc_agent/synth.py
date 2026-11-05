"""Synthetic network and alarm generator.

Builds a small optical + IP + power network, injects faults (fiber cuts, power
outages, line-card failures, planned works) on top of background noise, and
records the ground truth so every stage of the pipeline can be scored.

All names are fictional. The alarm patterns follow how these faults look on a
real NOC console: both ends of a cut span lose light, the IP links riding on it
go down a minute or two later, and an OTDR break-location alarm arrives several
minutes after that.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, timedelta

from .models import (
    Alarm,
    CardFailure,
    Dataset,
    FiberCut,
    IpLink,
    MaintenanceWindow,
    Node,
    PowerOutage,
    RootCause,
    Severity,
    Site,
    Source,
    Span,
    Topology,
    TrueIncident,
    World,
)

SITES = [
    ("ARD", "Ardena"), ("BEL", "Belmora"), ("COR", "Corvia"), ("DAL", "Dalmere"),
    ("ELV", "Elvaro"), ("FEN", "Fenwick"), ("GAR", "Garvoss"), ("HAL", "Halden"),
    ("IRO", "Irondale"), ("JUN", "Junera"), ("KES", "Kestrel"), ("LUM", "Lumen Bay"),
    ("MAR", "Marrow"), ("NOR", "Northgate"), ("OAK", "Oakhurst"), ("PIN", "Pinecrest"),
]  # fmt: skip

POWER_ALARMS = {"MAINS_FAILURE", "BATTERY_DISCHARGE", "LOW_BATTERY_VOLTAGE"}


def port_slot(port: str) -> int:
    """Router ports are named Gi0/<slot>/<port>."""
    return int(port.split("/")[1])


def build_topology(rng: random.Random, n_sites: int) -> Topology:
    if not 4 <= n_sites <= len(SITES):
        raise ValueError(f"n_sites must be between 4 and {len(SITES)}")
    codes = SITES[:n_sites]
    sites = [Site(code=c, name=n) for c, n in codes]
    nodes: list[Node] = []
    for code, _ in codes:
        lc = code.lower()
        nodes += [
            Node(name=f"otn-{lc}-01", site=code, kind="otn"),
            Node(name=f"rtr-{lc}-01", site=code, kind="router"),
            Node(name=f"pwr-{lc}-01", site=code, kind="power"),
        ]

    # a ring plus a few chords, like a regional backbone
    pairs = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    chords = [(i, j) for i in range(n_sites) for j in range(i + 2, n_sites) if (i, j) != (0, n_sites - 1)]
    rng.shuffle(chords)
    pairs += chords[: n_sites // 4]

    otn_ports: Counter[str] = Counter()
    rtr_ports: Counter[str] = Counter()
    spans: list[Span] = []
    links: list[IpLink] = []

    def next_rtr_port(code: str) -> str:
        idx = rtr_ports[code]
        rtr_ports[code] += 1
        return f"Gi0/{idx // 4}/{idx % 4}"

    for k, (i, j) in enumerate(pairs, start=1):
        a, z = codes[i][0], codes[j][0]
        cable_id = f"CBL-{k:03d}"
        length = round(rng.uniform(4, 60), 1)
        n_spans = 2 if rng.random() < 0.3 else 1
        for f in range(1, n_spans + 1):
            otn_ports[a] += 1
            otn_ports[z] += 1
            span = Span(
                span_id=f"SPN-{k:03d}-{f}",
                cable_id=cable_id,
                a_node=f"otn-{a.lower()}-01",
                a_port=f"OTS-{otn_ports[a]}",
                a_site=a,
                z_node=f"otn-{z.lower()}-01",
                z_port=f"OTS-{otn_ports[z]}",
                z_site=z,
                length_km=length,
            )
            spans.append(span)
            for _ in range(rng.choice([1, 1, 2])):
                links.append(
                    IpLink(
                        link_id=f"LNK-{len(links) + 1:03d}",
                        a_node=f"rtr-{a.lower()}-01",
                        a_port=next_rtr_port(a),
                        z_node=f"rtr-{z.lower()}-01",
                        z_port=next_rtr_port(z),
                        span_ids=[span.span_id],
                    )
                )
    return Topology(sites=sites, nodes=nodes, spans=spans, ip_links=links)


class _Builder:
    def __init__(self, rng: random.Random, topo: Topology):
        self.rng = rng
        self.topo = topo
        self.alarms: list[Alarm] = []

    def jitter(self, t: datetime, lo: float, hi: float) -> datetime:
        return t + timedelta(seconds=self.rng.uniform(lo, hi))

    def alarm(
        self,
        t: datetime,
        source: Source,
        node: str,
        name: str,
        severity: Severity,
        resource: str | None = None,
        span_id: str | None = None,
        **details: object,
    ) -> Alarm:
        site = self.topo.node(node).site
        a = Alarm(
            alarm_id=f"tmp-{len(self.alarms)}",
            raised_at=t.replace(microsecond=0),
            source=source,
            site=site,
            node=node,
            name=name,
            severity=severity,
            resource=resource,
            span_id=span_id,
            details={k: str(v) for k, v in details.items()},
        )
        self.alarms.append(a)
        return a


def _fiber_cut(b: _Builder, cable_id: str, t0: datetime, with_otdr: bool):
    spans = b.topo.spans_in_cable(cable_id)
    dist = round(b.rng.uniform(0.2, spans[0].length_km - 0.2), 2)
    out: list[Alarm] = []
    first_los = None
    for s in spans:
        los_a = b.alarm(b.jitter(t0, 0, 20), Source.OPTICAL, s.a_node, "OPTICAL_LOS",
                        Severity.CRITICAL, s.a_port, s.span_id)
        los_z = b.alarm(b.jitter(t0, 0, 20), Source.OPTICAL, s.z_node, "OPTICAL_LOS",
                        Severity.CRITICAL, s.z_port, s.span_id)
        first_los = first_los or los_a
        out += [los_a, los_z]
        for node, port in ((s.a_node, s.a_port), (s.z_node, s.z_port)):
            out.append(b.alarm(b.jitter(t0, 5, 40), Source.OPTICAL, node, "OSC_LOS",
                               Severity.MAJOR, port, s.span_id))
        for lk in b.topo.links_over_span(s.span_id):
            for node, port in ((lk.a_node, lk.a_port), (lk.z_node, lk.z_port)):
                out.append(b.alarm(b.jitter(t0, 20, 150), Source.IP, node, "LINK_DOWN",
                                   Severity.MAJOR, port))
    if with_otdr:
        s = spans[0]
        out.append(b.alarm(b.jitter(t0, 360, 840), Source.OPTICAL, s.a_node, "FIBER_BREAK_LOCATED",
                           Severity.CRITICAL, s.a_port, s.span_id, distance_km=f"{dist:.2f}",
                           related_alarm=first_los.alarm_id))
    return out, dist


def _power_outage(b: _Builder, site: str, t0: datetime, long: bool):
    lc = site.lower()
    pwr, otn, rtr = f"pwr-{lc}-01", f"otn-{lc}-01", f"rtr-{lc}-01"
    out = [
        b.alarm(t0, Source.POWER, pwr, "MAINS_FAILURE", Severity.CRITICAL, "AC-IN"),
        b.alarm(b.jitter(t0, 10, 90), Source.POWER, pwr, "BATTERY_DISCHARGE", Severity.MAJOR,
                "BATT", battery_voltage="52.1"),
    ]
    if long:
        # batteries run flat: the site goes dark and its neighbours lose light
        battery = round(b.rng.uniform(25, 50), 1)
        t_down = t0 + timedelta(minutes=battery)
        out.append(b.alarm(t_down - timedelta(minutes=6), Source.POWER, pwr, "LOW_BATTERY_VOLTAGE",
                           Severity.CRITICAL, "BATT", battery_voltage="46.2"))
        out.append(b.alarm(b.jitter(t_down, 0, 15), Source.OPTICAL, otn, "NODE_UNREACHABLE",
                           Severity.CRITICAL))
        out.append(b.alarm(b.jitter(t_down, 0, 15), Source.IP, rtr, "NODE_UNREACHABLE",
                           Severity.CRITICAL))
        for s in b.topo.spans_at_node(otn):
            node, port = (s.z_node, s.z_port) if s.a_node == otn else (s.a_node, s.a_port)
            out.append(b.alarm(b.jitter(t_down, 0, 30), Source.OPTICAL, node, "OPTICAL_LOS",
                               Severity.CRITICAL, port, s.span_id))
        for lk in b.topo.links_at_node(rtr):
            node, port = (lk.z_node, lk.z_port) if lk.a_node == rtr else (lk.a_node, lk.a_port)
            out.append(b.alarm(b.jitter(t_down, 10, 120), Source.IP, node, "LINK_DOWN",
                               Severity.MAJOR, port))
        end = t_down + timedelta(minutes=b.rng.uniform(20, 60))
    else:
        battery = round(b.rng.uniform(180, 300), 1)
        end = t0 + timedelta(minutes=b.rng.uniform(8, 20))
    return out, PowerOutage(site=site, start=t0, end=end, battery_minutes=battery)


def _card_failure(b: _Builder, node: str, t0: datetime):
    links = b.topo.links_at_node(node)

    def local_port(lk: IpLink) -> str:
        return lk.a_port if lk.a_node == node else lk.z_port

    slot = b.rng.choice(sorted({port_slot(local_port(lk)) for lk in links}))
    out = [b.alarm(t0, Source.IP, node, "CARD_FAILURE", Severity.CRITICAL, f"slot {slot}")]
    for lk in links:
        if port_slot(local_port(lk)) != slot:
            continue
        for n, p in ((lk.a_node, lk.a_port), (lk.z_node, lk.z_port)):
            out.append(b.alarm(b.jitter(t0, 0, 10), Source.IP, n, "LINK_DOWN", Severity.MAJOR, p))
    return out, CardFailure(node=node, slot=slot, start=t0)


def _noise(b: _Builder, t: datetime, world: World) -> None:
    rng, topo = b.rng, b.topo
    kind = rng.choices(["temp", "fan", "flap", "login", "rxlow", "lone_down"],
                       weights=[3, 2, 3, 2, 2, 1])[0]
    if kind == "temp":
        node = rng.choice([n.name for n in topo.nodes if n.kind != "router"])
        b.alarm(t, Source.POWER if node.startswith("pwr") else Source.OPTICAL, node,
                "HIGH_TEMPERATURE", Severity.MINOR, "sensor-1", temperature_c=rng.randint(46, 58))
    elif kind == "fan":
        node = rng.choice([n.name for n in topo.nodes if n.kind != "power"])
        b.alarm(t, Source.IP if node.startswith("rtr") else Source.OPTICAL, node,
                "FAN_SPEED_LOW", Severity.WARNING, "fan-2")
    elif kind == "login":
        node = rng.choice([n.name for n in topo.nodes if n.kind == "router"])
        b.alarm(t, Source.IP, node, "LOGIN_FAILURE", Severity.WARNING, "ssh", attempts=rng.randint(3, 9))
    elif kind == "rxlow":
        s = rng.choice(topo.spans)
        b.alarm(t, Source.OPTICAL, s.a_node, "OPTICAL_POWER_LOW", Severity.MINOR, s.a_port, s.span_id,
                rx_dbm=f"{rng.uniform(-24, -19):.1f}")
    else:
        lk = rng.choice(topo.ip_links)
        node, port = (lk.a_node, lk.a_port) if rng.random() < 0.5 else (lk.z_node, lk.z_port)
        if kind == "flap":
            b.alarm(t, Source.IP, node, "LINK_FLAP", Severity.WARNING, port,
                    flaps_last_hour=rng.randint(2, 6))
        else:
            # a single LINK_DOWN on a port that has been flapping all day
            b.alarm(t, Source.IP, node, "LINK_DOWN", Severity.MAJOR, port)
            hist = world.history.setdefault(node, {})
            hist["LINK_FLAP"] = hist.get("LINK_FLAP", 0) + rng.randint(12, 30)


def generate_dataset(
    seed: int = 7,
    n_sites: int = 12,
    hours: float = 6,
    fiber_cuts: int = 3,
    power_outages: int = 2,
    card_failures: int = 1,
    planned_works: int = 1,
    noise: int = 40,
) -> Dataset:
    rng = random.Random(seed)
    topo = build_topology(rng, n_sites)
    start = datetime(2026, 3, 2, 6, 0) + timedelta(days=seed % 28)
    end = start + timedelta(hours=hours)
    b = _Builder(rng, topo)
    world = World(history={
        n.name: {"LOGIN_FAILURE": rng.randint(0, 3), "HIGH_TEMPERATURE": rng.randint(0, 2)}
        for n in topo.nodes
    })
    truth: list[tuple[RootCause, str, list[Alarm], datetime, float | None]] = []

    def when() -> datetime:
        return start + timedelta(minutes=rng.uniform(20, hours * 60 - 70))

    cables = sorted({s.cable_id for s in topo.spans})
    rng.shuffle(cables)
    touched: set[str] = set()

    for _ in range(fiber_cuts):
        cable, t0 = cables.pop(), when()
        alarms, dist = _fiber_cut(b, cable, t0, with_otdr=True)
        world.cuts.append(FiberCut(cable_id=cable, start=t0, break_km=dist))
        truth.append((RootCause.FIBER_CUT, cable, alarms, t0, dist))
        span = topo.spans_in_cable(cable)[0]
        touched |= {span.a_site, span.z_site}

    for i in range(planned_works):
        cable, t0 = cables.pop(), when()
        alarms, dist = _fiber_cut(b, cable, t0, with_otdr=False)
        world.cuts.append(FiberCut(cable_id=cable, start=t0, end=t0 + timedelta(minutes=90),
                                   break_km=dist))
        world.maintenance.append(MaintenanceWindow(
            window_id=f"MW-{i + 1:03d}", target=cable, start=t0 - timedelta(minutes=20),
            end=t0 + timedelta(hours=2), description=f"Planned splicing works on {cable}"))
        truth.append((RootCause.PLANNED_WORK, cable, alarms, t0, None))
        span = topo.spans_in_cable(cable)[0]
        touched |= {span.a_site, span.z_site}

    # a maintenance window that overlaps nothing, so "any window exists" is not a shortcut
    if cables:
        t = when()
        world.maintenance.append(MaintenanceWindow(
            window_id="MW-900", target=cables[-1], start=t, end=t + timedelta(hours=1),
            description=f"Cable inspection on {cables[-1]}"))

    free_sites = [c for c, _ in SITES[:n_sites] if c not in touched] or [c for c, _ in SITES[:n_sites]]
    rng.shuffle(free_sites)
    for i in range(power_outages):
        site = free_sites[i % len(free_sites)]
        t0 = when()
        if i == 0 and truth:
            # start alongside the first fiber cut: a pure time-window grouper merges the two
            t0 = truth[0][3] + timedelta(seconds=rng.uniform(-60, 60))
        alarms, outage = _power_outage(b, site, t0, long=(i == 0))
        world.outages.append(outage)
        truth.append((RootCause.POWER_LOSS, site, alarms, t0, None))

    routers = [n.name for n in topo.nodes if n.kind == "router" and topo.links_at_node(n.name)]
    for _ in range(card_failures):
        node, t0 = rng.choice(routers), when()
        alarms, failure = _card_failure(b, node, t0)
        world.card_failures.append(failure)
        truth.append((RootCause.HARDWARE_FAILURE, node, alarms, t0, None))

    for _ in range(noise):
        _noise(b, start + timedelta(minutes=rng.uniform(0, hours * 60)), world)

    # number alarms in time order, the way a monitoring system would
    b.alarms.sort(key=lambda a: a.raised_at)
    new_id = {a.alarm_id: f"ALM-{i:05d}" for i, a in enumerate(b.alarms, start=1)}
    for a in b.alarms:
        a.alarm_id = new_id[a.alarm_id]
        if "related_alarm" in a.details:
            a.details["related_alarm"] = new_id[a.details["related_alarm"]]

    truth.sort(key=lambda t: t[3])
    incidents = [
        TrueIncident(
            incident_id=f"INC-{i:02d}",
            root_cause=cause,
            location=location,
            alarm_ids=sorted(a.alarm_id for a in alarms),
            started_at=t0.replace(microsecond=0),
            break_km=dist,
        )
        for i, (cause, location, alarms, t0, dist) in enumerate(truth, start=1)
    ]
    return Dataset(seed=seed, start=start, end=end, topology=topo, world=world,
                   alarms=b.alarms, incidents=incidents)
