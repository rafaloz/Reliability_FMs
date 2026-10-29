"""Generate the main transfer table and related supplementary contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Analyses').is_dir())
ORDER=('yaware','anatcl','brainsegfounder','3dneurosimclr','neurovfm','curia2','brainiac')

def number(value,signed=False):
    if not np.isfinite(value):raise ValueError('Nonfinite table value')
    if abs(value)<.005:return '0.00'
    return f'{value:+.2f}' if signed else f'{value:.2f}'

def interval(record,signed=False):
    return '$'+number(record['estimate'],signed)+r'\;['+number(record['ci_lower'])+', '+number(record['ci_upper'])+']$'

def probability(value,star=False):
    if not 0<=value<=1:raise ValueError('Invalid probability')
    return f'${value:.3f}'+(r'^{*}' if star and value<.05 else '')+'$'

def finish(lines):return '\n'.join([*lines,r'\botrule',r'\end{tabular}',''])

def build(figure2,figure3,figure4,output):
    inputs=[Path(figure2),Path(figure3),Path(figure4)]
    payloads=[json.loads(p.read_text()) for p in inputs]
    embeddings,frozen,adapted=payloads
    if set(frozen['conditions'])!=set(ORDER) or set(adapted['conditions'])!=set(ORDER):
        raise ValueError('Expected the same seven model families')
    for condition in ('anatcl','matched_simclr'):
        if embeddings['conditions'][condition]['primary_folds']!=[0]:
            raise ValueError('Expected fold0 for both primary embedding conditions')
    main=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Model & Frozen & Fine-tuned & Gain & $p$ & $q_{\mathrm{BH}}$ \\',r'\midrule']
    endpoints=[('mae','accuracy_gain','Internal-test MAE (years)','internal_test_mae'),
               ('dbicc','reliability_gain','ON-Harmony predicted-age dbICC','prediction_dbicc'),
               ('scanner_difference','stability_gain','ON-Harmony mean absolute pairwise difference (years)','cross_scanner_absolute_difference')]
    for i,(metric,endpoint,title,frozen_key) in enumerate(endpoints):
        if i:main.append(r'\midrule')
        main.append(r'\multicolumn{6}{l}{\textbf{'+title+r'}} \\')
        for condition in ORDER:
            record=adapted['conditions'][condition]
            if len(record['primary_runs'])!=3:raise ValueError('Expected three primary seeds per model')
            baseline=record['absolute']['frozen'][metric]
            reference=frozen['conditions'][condition][frozen_key]
            np.testing.assert_allclose([baseline[k] for k in ('estimate','ci_lower','ci_upper')],[reference[k] for k in ('estimate','ci_lower','ci_upper')],rtol=0,atol=1e-10)
            gain=record['primary_gains'][endpoint]
            main.append(' & '.join([record['label'],interval(baseline),interval(record['absolute']['adapted'][metric]),interval(gain,True),probability(gain['p_value_two_sided']),probability(gain['q_value_bh_across_21_contrasts'],True)])+r' \\')
    comparison=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Model & Embedding dbICC & Ridge dbICC & Difference [95\% CI] & $p$ & $q_{\mathrm{BH}}$ \\',r'\midrule']
    for condition in ORDER:
        record=frozen['conditions'][condition];gain=record['dbicc_change']
        comparison.append(' & '.join([record['label'],'$'+number(record['embedding_dbicc']['estimate'])+'$','$'+number(record['prediction_dbicc']['estimate'])+'$',interval(gain,True),probability(gain['p_value_two_sided']),probability(gain['q_value_bh'],True)])+r' \\')
    contrasts=embeddings.get('targeted_contrasts',embeddings.get('contrasts'))
    control=[r'\begin{tabular}{lcccc}',r'\toprule',r'Analysis & AnatCL dbICC & AnatCL--SimCLR dbICC & Difference [95\% CI] & $p$ \\',r'\midrule']
    for label,key,sensitivity in [('Fold 0','anatcl_minus_matched_simclr',False),('Five-fold sensitivity','anatcl_minus_matched_simclr_all_folds',True)]:
        records=[embeddings['conditions'][c] for c in ('anatcl','matched_simclr')]
        values=[r['fold_sensitivity']['dbicc'] if sensitivity else r['dbicc'] for r in records]
        contrast=contrasts[key]
        np.testing.assert_allclose(contrast['estimate'],values[0]['estimate']-values[1]['estimate'],rtol=0,atol=1e-12)
        control.append(' & '.join([label,interval(values[0]),interval(values[1]),interval(contrast,True),probability(contrast['p_value_two_sided'])])+r' \\')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    outputs={}
    for name,lines in [('table4_brain_age_summary.tex',main),('supp_table_embedding_prediction_dbicc.tex',comparison),('supp_table_anatcl_checkpoint_sensitivity.tex',control)]:
        path=output/name;path.write_text(finish(lines));outputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'status':'PASS','primary_checkpoint':'fold0 for AnatCL and AnatCL-SimCLR','adapted_primary_runs_per_model':3,
              'inputs':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'outputs':outputs,
              'rounding':{'MAE_years':2,'dbICC':2,'differences':2,'p_q':3},'significance':'unrounded global q<0.05'}
    (output/'brain_age_results_tables.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS: generated main Table4, S5 and checkpoint-sensitivity table')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure2-results',type=Path,default=ROOT/'Results/Figure2/figure2_reliability_results.json')
    p.add_argument('--figure3-results',type=Path,default=ROOT/'Results/Figure3/figure3_brain_age_results.json')
    p.add_argument('--figure4-results',type=Path,default=ROOT/'Results/Figure4/figure4_adaptation_results.json')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();build(a.figure2_results,a.figure3_results,a.figure4_results,a.output_dir)
