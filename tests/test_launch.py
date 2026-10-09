"""strl launch: race every capacity route, keep the first that starts, cancel and ledger the rest (fake ScaleTrain, no network)."""
import os
import tempfile
import unittest
from pathlib import Path

from strl import launch as L


class FakeScaleTrain:
    """Scripted job states: script[route_code] = list of (status, detail) per poll interval (state advances when the launcher sleeps, not
    when it reads, like a real job); the last entry repeats. Cancel -> CANCELED."""

    def __init__(self, script, started=()):
        self.script, self.started = script, set(started)
        self.jobs, self.polls, self.cancelled, self.staged, self.recorded, self.now, self.ticks = {}, {}, set(), [], [], 0.0, 0

    def api(self):
        return L.Api(stage=self.stage, submit=self.submit, get=self.get, cancel=self.cancel, record=self.record,
                     pods_started=lambda experiment, attempt: attempt in self.started, clock=lambda: self.now, sleep=self.sleep, log=lambda m: None)

    def stage(self, experiment, *, attempt, mode, team, allow_borrowing, overrides, reservation=None):
        self.staged.append({'attempt': attempt, 'team': team, 'borrow': allow_borrowing, 'overrides': list(overrides)})
        return {'job_file': f'evidence/{attempt}/job.json'}

    def submit(self, job_file):
        attempt = Path(job_file).parent.name
        job_id = f'job-{attempt}'
        self.jobs[job_id] = attempt
        self.polls[job_id] = self.ticks
        return job_id

    def get(self, job_id):
        if job_id in self.cancelled:
            return {'status': 'CANCELED', 'status_detail': 'absent'}
        attempt = self.jobs[job_id]
        script = self.script.get(attempt.rsplit('-', 1)[-1] if attempt.rsplit('-', 1)[-1] in self.script else attempt, [('QUEUED', 'waiting')])
        i = min(self.ticks - self.polls[job_id], len(script) - 1)
        status, detail = script[i]
        return {'status': status, 'status_detail': detail}

    def cancel(self, job_id):
        self.cancelled.add(job_id)

    def record(self, experiment, attempt, job_id, never_started=False, terminal_at=None, allocated_floor=None):
        self.floors = getattr(self, 'floors', {})
        self.floors[attempt] = allocated_floor
        self.recorded.append((attempt, never_started))
        self.terminal_at = getattr(self, 'terminal_at', {})
        self.terminal_at[attempt] = terminal_at
        return {'ok': True}

    def sleep(self, seconds):
        self.now += seconds
        self.ticks += 1


class Race(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_first_route_to_start_wins_and_the_rest_are_cancelled_uncharged(self):
        fake = FakeScaleTrain({'rb': [('QUEUED', 'w'), ('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]})
        out = L.race('exp.yaml', 'run1', 'train', fake.api(), poll=10)
        self.assertEqual(out['winner'], 'run1-rb')
        self.assertEqual(fake.cancelled, {'job-run1-fb', 'job-run1-ro', 'job-run1-fo'})
        self.assertEqual(sorted(fake.recorded), [('run1-fb', True), ('run1-fo', True), ('run1-ro', True)])   # no pod ran: not charged
        self.assertNotIn('job-run1-rb', fake.cancelled)

    def test_routes_stage_team_and_borrowing(self):
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]})
        L.race('exp.yaml', 'run1', 'canary', fake.api(), overrides=['compute.nodes=1'], poll=1)
        by = {s['attempt']: s for s in fake.staged}
        self.assertEqual((by['run1-fb']['team'], by['run1-fb']['borrow']), (L.FOUNDATION, True))
        self.assertEqual((by['run1-ro']['team'], by['run1-ro']['borrow']), (L.RESEARCH, False))
        self.assertIn('compute.allow_borrowing=false', by['run1-fo']['overrides'])     # base/infra.yaml defaults borrowing on
        self.assertEqual(by['run1-fb']['overrides'], ['compute.nodes=1'])

    def test_two_routes_starting_in_one_poll_keep_one_and_charge_the_other(self):
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')], 'rb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]}, started={'run1-rb'})
        out = L.race('exp.yaml', 'run1', 'train', fake.api(), poll=5)
        self.assertEqual(out['winner'], 'run1-fb')                     # the runbook's preferred route wins a tie
        self.assertIn(('run1-rb', False), fake.recorded)               # its pod ran: charged (never_started False)
        self.assertTrue(next(l for l in out['losers'] if l['attempt'] == 'run1-rb')['charged'])

    def test_routes_after_a_started_one_are_not_submitted(self):
        fake = FakeScaleTrain({'fb': [('IN_PROGRESS', 'absent')]})
        out = L.race('exp.yaml', 'run1', 'train', fake.api(), poll=5)
        self.assertEqual(out['winner'], 'run1-fb')
        self.assertEqual([x['attempt'] for x in fake.staged], ['run1-fb'])     # nothing else staged, submitted, cancelled or charged
        self.assertEqual(fake.recorded, [])

    def test_a_loser_seen_allocated_is_charged_even_before_its_pod_leaves_a_trace(self):
        # measured: v1phi-rb went IN_PROGRESS in the winner's poll, was cancelled, and its pod wrote budget.json 89 s AFTER CANCELED
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')], 'rb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]}, started=())
        out = L.race('exp.yaml', 'run1', 'train', fake.api(), poll=5)
        self.assertEqual(out['winner'], 'run1-fb')
        self.assertIn(('run1-rb', False), fake.recorded)               # charged: it was allocated
        self.assertIsNotNone(fake.terminal_at['run1-rb'])              # through the cancel tail, not the API's CANCELED time
        self.assertIn(('run1-ro', True), fake.recorded)                # only ever QUEUED: not charged
        self.assertIsNone(fake.terminal_at['run1-ro'])

    def test_no_capacity_within_the_window_cancels_everything(self):
        fake = FakeScaleTrain({})
        out = L.race('exp.yaml', 'run1', 'train', fake.api(), poll=60, max_wait_seconds=600)
        self.assertIsNone(out['winner'])
        self.assertEqual(len(fake.cancelled), 4)
        self.assertEqual(len(fake.recorded), 4)

    def test_a_job_that_started_and_finished_between_polls_still_wins(self):
        fake = FakeScaleTrain({'fb': [('COMPLETED', 'absent')], 'rb': [('FAILED', 'absent')]}, started={'run1-fb'})
        out = L.race('exp.yaml', 'run1', 'dry', fake.api(), poll=5)
        self.assertEqual(out['winner'], 'run1-fb')
        self.assertNotIn('job-run1-fb', fake.cancelled)

    def test_route_selection_and_name_length(self):
        self.assertEqual([r[0] for r in L.select_routes('ro,fb')], ['ro', 'fb'])
        with self.assertRaises(ValueError):
            L.select_routes('xx')
        with self.assertRaises(ValueError):
            L.race('exp.yaml', 'a' * 27, 'train', FakeScaleTrain({}).api())

    def test_every_name_a_launch_may_create_is_checked_up_front(self):
        L.check_names('a' * 22, relaunch_on_preempt=2, fallback=False)              # 22 + '-p2' = 25
        with self.assertRaises(ValueError):
            L.check_names('a' * 22, relaunch_on_preempt=2, fallback=True)            # 22 + '-alt' + '-p2' = 29
        L.check_names('a' * 19, relaunch_on_preempt=2, fallback=True)
        self.assertFalse(L.FALLBACK_SUFFIX.lstrip('-').startswith(tuple(c for c, _, _ in L.ROUTES)))

    def test_preemption_is_detected_cancelled_recorded_and_relaunched(self):
        fake = FakeScaleTrain({'fb': [('IN_PROGRESS', 'absent'), ('IN_PROGRESS', 'absent'), ('QUEUED', 'preemption_reported')],
                               'run1-p1-fb': [('IN_PROGRESS', 'absent'), ('COMPLETED', 'absent')]}, started={'run1-fb'})
        # the relaunch race uses the same route codes; give its fb a different script by attempt name
        fake.script['p1-fb'] = fake.script.pop('run1-p1-fb')
        original_get = fake.get

        def get(job_id):
            attempt = fake.jobs.get(job_id, '')
            if attempt.startswith('run1-p1-') and job_id not in fake.cancelled:
                script = fake.script['p1-fb'] if attempt.endswith('-fb') else [('QUEUED', 'w')]
                i = min(fake.ticks - fake.polls[job_id], len(script) - 1)
                return {'status': script[i][0], 'status_detail': script[i][1]}
            return original_get(job_id)
        api = fake.api()
        api.get = get
        history = L.launch('exp.yaml', 'run1', 'train', api, poll=1, follow=True, relaunch_on_preempt=1, resume_on_preempt=True)
        self.assertEqual([h['winner'] for h in history], ['run1-fb', 'run1-p1-fb'])
        self.assertEqual(history[0]['outcome']['outcome'], 'preempted')
        self.assertEqual(history[1]['outcome']['outcome'], 'terminal')
        self.assertIn(('run1-fb', False), fake.recorded)               # the preempted attempt is charged (its pod ran)
        self.assertTrue(Path('evidence/run1/launch.json').is_file())
        relaunched = [x for x in fake.staged if x['attempt'].startswith('run1-p1-')]
        self.assertTrue(all('run.resume_from=run1-fb' in x['overrides'] for x in relaunched))   # continues the preempted attempt


class ReviewFindings(unittest.TestCase):
    """Independent review of the launcher (2026-10-09): accounting and failure paths."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_a_loser_charged_from_first_allocation_not_from_its_queue_wait(self):
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w')] * 5 + [('IN_PROGRESS', 'absent')], 'rb': [('QUEUED', 'w')] * 5 + [('IN_PROGRESS', 'absent')]})
        L.race('exp.yaml', 'run1', 'train', fake.api(), poll=60)
        self.assertIn(('run1-rb', False), fake.recorded)
        self.assertGreaterEqual(fake.floors['run1-rb'], 4 * 60)        # first seen allocated after 5 polls, minus one poll: not created_at

    def test_a_loser_seen_running_only_while_settling_is_charged(self):
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')], 'rb': [('QUEUED', 'w'), ('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]})
        original_cancel = fake.cancel
        calls = []

        def slow_cancel(job_id):      # the first cancel does not take effect at once: rb starts before it ends
            calls.append(job_id)
            if calls.count(job_id) > 2:
                original_cancel(job_id)
        api = fake.api()
        api.cancel = slow_cancel
        L.race('exp.yaml', 'run1', 'train', api, poll=5)
        self.assertIn(('run1-rb', False), fake.recorded)

    def test_a_queued_job_that_fails_without_a_pod_is_not_charged(self):
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')], 'rb': [('FAILED', 'admission')]})
        L.race('exp.yaml', 'run1', 'train', fake.api(), poll=5)
        self.assertIn(('run1-rb', True), fake.recorded)

    def test_an_error_while_submitting_cancels_the_routes_already_submitted(self):
        fake = FakeScaleTrain({})
        api = fake.api()
        original = api.stage

        def stage(experiment, **kw):
            if kw['attempt'].endswith('-ro'):
                raise OSError('S3 upload failed')
            return original(experiment, **kw)
        api.stage = stage
        with self.assertRaises(OSError):
            L.race('exp.yaml', 'run1', 'train', api, poll=5)
        self.assertEqual(fake.cancelled, {'job-run1-fb', 'job-run1-rb'})

    def test_unknown_pod_evidence_never_crowns_a_winner(self):
        fake = FakeScaleTrain({'fb': [('FAILED', 'x')], 'rb': [('QUEUED', 'w'), ('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]})
        api = fake.api()
        api.pods_started = lambda experiment, attempt: None if attempt.endswith('-fb') else False
        out = L.race('exp.yaml', 'run1', 'train', api, poll=5)
        self.assertEqual(out['winner'], 'run1-rb')
        self.assertIn(('run1-fb', False), fake.recorded)               # unknown is charged (fail closed), but it did not win

    def test_a_ledger_failure_is_reported_not_swallowed(self):
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]})
        api = fake.api()

        def broken(*a, **k):
            raise RuntimeError('scale-train get failed')
        api.record = broken
        out = L.race('exp.yaml', 'run1', 'train', api, poll=5)
        self.assertTrue(all(l['ledger'].get('unrecorded') for l in out['losers']))

    def test_a_relaunch_resumes_from_the_latest_attempt_with_a_checkpoint(self):
        script = {'fb': [('IN_PROGRESS', 'absent'), ('QUEUED', 'preemption_reported')]}
        fake = FakeScaleTrain(script, started={'run1-fb', 'run1-p1-fb'})
        api = fake.api()
        api.has_checkpoint = lambda experiment, attempt: attempt == 'run1-fb'     # run1-p1-fb was preempted before its first save
        history = L.launch('exp.yaml', 'run1', 'train', api, poll=1, follow=True, relaunch_on_preempt=2, resume_on_preempt=True)
        self.assertEqual([h['winner'] for h in history], ['run1-fb', 'run1-p1-fb', 'run1-p2-fb'])
        p2 = [x for x in fake.staged if x['attempt'] == 'run1-p2-fb'][0]
        self.assertIn('run.resume_from=run1-fb', p2['overrides'])


class SecondReview(unittest.TestCase):
    """Re-review of the launcher fixes (2026-10-09)."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_a_trace_appearing_after_canceled_is_charged(self):
        fake = FakeScaleTrain({'fb': [('QUEUED', 'w'), ('IN_PROGRESS', 'absent')]})
        api = fake.api()
        cancelled_at = {}
        original_cancel = fake.cancel

        def cancel(job_id):
            cancelled_at.setdefault(job_id, fake.now)
            original_cancel(job_id)
        api.cancel = cancel
        # rb's pod writes budget.json 90 s after it was cancelled (never seen allocated by the launcher)
        api.pods_started = lambda experiment, attempt: attempt.endswith('-rb') and fake.now >= cancelled_at.get('job-run1-rb', 1e9) + 90
        L.race('exp.yaml', 'run1', 'train', api, poll=15)
        self.assertIn(('run1-rb', False), fake.recorded)
        self.assertIn(('run1-ro', True), fake.recorded)

    def test_unknown_evidence_forever_ends_the_race(self):
        fake = FakeScaleTrain({'fb': [('FAILED', 'image pull')], 'rb': [('FAILED', 'image pull')]})
        api = fake.api()
        api.pods_started = lambda experiment, attempt: None
        out = L.race('exp.yaml', 'run1', 'train', api, routes='fb,rb', poll=5)
        self.assertIsNone(out['winner'])
        self.assertTrue(all(not never for _, never in fake.recorded))     # unknown is charged

    def test_an_interrupt_while_waiting_cancels_every_route(self):
        fake = FakeScaleTrain({})
        api = fake.api()
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:
                raise KeyboardInterrupt
            fake.sleep(seconds)
        api.sleep = sleep
        with self.assertRaises(KeyboardInterrupt):
            L.race('exp.yaml', 'run1', 'train', api, poll=5)
        self.assertEqual(len(fake.cancelled), 4)

    def test_an_unsettled_preempted_job_stops_the_relaunch_and_is_a_problem(self):
        script = {'fb': [('IN_PROGRESS', 'absent'), ('QUEUED', 'preemption_reported')]}
        fake = FakeScaleTrain(script, started={'run1-fb'})
        api = fake.api()
        api.cancel = lambda job_id: None                                       # the requeued job never leaves QUEUED
        history = L.launch('exp.yaml', 'run1', 'train', api, poll=1, follow=True, relaunch_on_preempt=2)
        self.assertEqual(len(history), 1)
        self.assertTrue(L._problems(history))

    def test_a_cancelled_spelling_never_breaks_the_budget(self):
        from strl.budget import remaining_seconds
        self.assertEqual(remaining_seconds([{'job_id': 'j', 'status': 'CANCELLED', 'allocated_at': None, 'terminal_at': 'x'}], 100), 100)


if __name__ == '__main__':
    unittest.main()
