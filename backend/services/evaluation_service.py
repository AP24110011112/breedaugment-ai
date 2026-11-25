"""Evaluation service wrapper."""
from backend.config import REPORTS
from backend.ml.evaluate import evaluate_all
from backend.utils.helpers import read_json

def get_results():
    saved=read_json(REPORTS/'evaluation_results.json',None)
    if saved: return saved
    # Preserve a completed per-model report if a prior comparison stopped
    # before it could write the combined results file.
    models=[report for name in ('original','augmented')
            if (report:=read_json(REPORTS/f'{name}_evaluation.json',None))]
    keys=('model','validation_accuracy','validation_loss','validation_samples',
          'accuracy','precision','recall','f1_score','test_samples')
    return {'pending':len(models)<2,'models':models,
            'comparison':[{key:report.get(key) for key in keys} for report in models],
            'errors':[]}

def run_evaluation(): return evaluate_all()
