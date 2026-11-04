"""Run configured training, evaluation, and reporting on prepared MLRan data."""
from src import check_raw, evaluate, make_report, quality_checks
from src.common import Paths
from src.exp.run_all import main as experiments


def main():
    paths = Paths.default()
    check_raw.verify(paths, full_hash=True)
    if not all(check.passed for check in quality_checks.run_all(paths, write=False)):
        raise RuntimeError('Data quality checks failed; run make data before reproducing')
    experiments()
    evaluate.run()
    make_report.run()


if __name__ == '__main__':
    main()
