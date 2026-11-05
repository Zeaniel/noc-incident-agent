from noc_agent.models import Dataset, RootCause
from noc_agent.synth import generate_dataset


def test_same_seed_same_data():
    a, b = generate_dataset(seed=3), generate_dataset(seed=3)
    assert a.model_dump() == b.model_dump()
    assert generate_dataset(seed=4).model_dump() != a.model_dump()


def test_every_scenario_is_present(ds):
    causes = {i.root_cause for i in ds.incidents}
    assert causes == {RootCause.FIBER_CUT, RootCause.POWER_LOSS, RootCause.HARDWARE_FAILURE,
                      RootCause.PLANNED_WORK}


def test_alarms_reference_real_inventory(ds):
    ids = {a.alarm_id for a in ds.alarms}
    for a in ds.alarms:
        assert ds.topology.node(a.node) is not None
        assert ds.topology.node(a.node).site == a.site
        if a.span_id:
            assert ds.topology.span(a.span_id) is not None
        if "related_alarm" in a.details:
            assert a.details["related_alarm"] in ids
    for inc in ds.incidents:
        assert set(inc.alarm_ids) <= ids


def test_alarm_ids_follow_time_order(ds):
    times = [a.raised_at for a in sorted(ds.alarms, key=lambda a: a.alarm_id)]
    assert times == sorted(times)


def test_fiber_cut_shape(ds):
    for inc in ds.incidents:
        if inc.root_cause != RootCause.FIBER_CUT:
            continue
        alarms = [a for a in ds.alarms if a.alarm_id in inc.alarm_ids]
        names = [a.name for a in alarms]
        assert names.count("OPTICAL_LOS") >= 2  # both ends lose light
        otdr = next(a for a in alarms if a.name == "FIBER_BREAK_LOCATED")
        los = next(a for a in alarms if a.alarm_id == otdr.details["related_alarm"])
        assert los.name == "OPTICAL_LOS"
        assert 300 <= (otdr.raised_at - los.raised_at).total_seconds() <= 900
        assert float(otdr.details["distance_km"]) == inc.break_km


def test_save_and_load(tmp_path, ds):
    ds.save(tmp_path)
    assert Dataset.load(tmp_path).model_dump() == ds.model_dump()
    assert len((tmp_path / "alarms.jsonl").read_text().splitlines()) == len(ds.alarms)
