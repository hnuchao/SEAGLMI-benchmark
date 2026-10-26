"""Regenerate the benchmark result tables and figures from the aggregate results.

No fitting, scoring, resampling or model selection is performed.
usage: python scripts/rebuild_tables_figures.py   (writes to generated/)
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, io, json, math, shutil, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

PAPER=Path(__file__).resolve().parents[1]
PROJECT=PAPER
DATA=PAPER/'results'
TABLE=PAPER/'generated/tables'
FIG=PAPER/'generated/figures'
METRICS=['AUROC','AP','MCC','precision','recall','F1']
METHODS=['seaglmi','sitegbdt','pairfeat_gbdt','deeplmi','spgnn','hetgat_lmi_v2','mpgk_lmi','tec_lncmir','premli_keras224_v2','slnpm_correct','ndalma','kmer_lr','seed_sites','degree_prior']
LABELS=dict(zip(METHODS,['SEAGLMI','SEAGLMI-GBDT','PairFeat-GBDT','DeepLMI','SPGNN','HetGAT-LMI','MPGK-LMI','TEC-LncMir','preMLI','SLNPM-SC','NDALMA','k-mer LR','Seed rule','Degree prior']))
SETTINGS=['warm','blind_lncRNA','blind_miRNA','blind_both']
SLABEL={'warm':'Inherited warm split','blind_lncRNA':'Blind-lncRNA','blind_miRNA':'Blind-miRNA','blind_both':'Blind-both','source':'Source-record proxy'}
CONTRASTS=['eligibility_fixed','eligibility_dynamic','dynamic_base','dynamic_all','ratio_two_minus_one','eligibility_by_sampling_interaction']
CLABEL={'eligibility_fixed':'Eligibility: fixed','eligibility_dynamic':'Eligibility: dynamic','dynamic_base':'Dynamic: base','dynamic_all':'Dynamic: all','ratio_two_minus_one':'2:1 minus 1:1','eligibility_by_sampling_interaction':'Eligibility x sampling'}
ORANGE='#D95A32'; DARK='#8C2D23'; BLUE='#536B7A'; LIGHT='#D8AC90'
INPUTS={}; CHECKS={}; PLOT_ROWS=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def readj(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def ensure(x,message):
    if not x:raise AssertionError(message)

def close(x,y):ensure(math.isclose(float(x),float(y),abs_tol=2e-12,rel_tol=2e-12),('Frozen aggregate mismatch',x,y))

def fnum(v):return f'{float(v):.3f}'

def pm(r,k):return f"{float(r[k+'_mean']):.3f} $\\pm$ {float(r[k+'_std']):.3f}"

def esc(s):return str(s).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')

def write_rows(name,rs,fields=None):
    ensure(rs,'Empty verified export '+name)
    with (DATA/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rs[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rs)

def main_table(phase,protocol=None):
    if phase=='A':
        label='strict' if protocol=='strict' else 'standard'
        caption=(f'{protocol.capitalize()} protocol, 1:1 outer evaluation. AUROC/AP are equal means over five seed-0 folds; '
                 'all 14 methods are shown. Fold SDs and the six metrics at both evaluation ratios are reported in the supplement. '
                 'All supervised baseline training uses the frozen fixed 1:2 rows. Cross-protocol differences include the frozen context pipeline and are not a sampling-only effect.')
        header=['Method']+[x for setting in SETTINGS for x in ['AUROC','AP']]
        text=[r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{l'+'c'*8+'}',r'\toprule',r' & '+ ' & '.join(r'\multicolumn{2}{c}{'+SLABEL[s]+'}' for s in SETTINGS)+r' \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}', ' & '.join(header)+r' \\',r'\midrule']
        for m in METHODS:
            cells=[LABELS[m]]
            for s in SETTINGS:
                r=A_LOOK[m,protocol,s,'1'];cells.extend(fnum(r[k+'_mean']) for k in ['AUROC','AP'])
            text.append(' & '.join(cells)+r' \\')
    else:
        label='source-results';caption=('Source-record proxy holdout, 1:1 evaluation. Each value averages five folds equally within each of five frozen negative draws, then averages the draw means equally. '
             'This is not independent biological confirmation. The original conditional 2000-replicate source-record bootstrap intervals are shown in the supplement and the source interval figure.')
        text=[r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\small',r'\setlength{\tabcolsep}{8pt}',r'\begin{tabular}{lcccc}',r'\toprule',r' & \multicolumn{2}{c}{Strict} & \multicolumn{2}{c}{Standard} \\',r'Method & AUROC & AP & AUROC & AP \\',r'\midrule']
        for m in METHODS:
            cells=[LABELS[m]]
            for protocol in ['strict','standard']:cells.extend(fnum(C_LOOK[m,protocol,'1'][k+'_mean']) for k in ['AUROC','AP'])
            text.append(' & '.join(cells)+r' \\')
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (TABLE/('tab_screened_'+(label if phase=='A' else 'source')+'.tex')).write_text('\n'.join(text)+'\n',encoding='utf-8')

def longtable(name,label,caption,headers,body,landscape=False):
    n=len(headers);fmt='l'*n
    text=([r'\begin{landscape}'] if landscape else [])+[r'{\scriptsize',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.08}',r'\begin{longtable}{'+fmt+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endfirsthead',r'\multicolumn{'+str(n)+r'}{l}{\textit{Continued from previous page}}\\',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{'+str(n)+r'}{r}{\textit{Continued on next page}}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot']+body+[r'\end{longtable}',r'}']+([r'\end{landscape}'] if landscape else [])
    (TABLE/('tab_screened_'+name+'.tex')).write_text('\n'.join(text)+'\n',encoding='utf-8')

def group_title(title,n):return r'\multicolumn{'+str(n)+r'}{l}{\textit{'+esc(title)+r'}}\\'

def supplementary_tables():
    body=[]
    for protocol in ['strict','standard']:
        for s in SETTINGS:
            for ratio in ['1','10']:
                body.append(group_title(f'{protocol.capitalize()} | {SLABEL[s]} | outer 1:{ratio}',7))
                for m in METHODS:body.append(' & '.join([LABELS[m]]+[pm(A_LOOK[m,protocol,s,ratio],k) for k in METRICS])+r' \\')
    longtable('A_full','tab:screened-A-full','All A results: six equal-fold means $\\pm$ sample SD over five seed-0 folds. Both frozen outer ratios are reported. Inherited warm split does not guarantee that both endpoints appeared in training. No significance tests are reported.',['Method']+METRICS,body,True)
    body=[]
    for protocol in ['strict','standard']:
        for s in ['warm','blind_both']:
            for ratio in ['1','10']:
                body.append(group_title(f'{protocol.capitalize()} | {SLABEL[s]} | outer 1:{ratio}',8))
                for m in METHODS:
                    for cell in ['F-base-2','F-all-2','D-base-2','D-all-2','F-base-1']:
                        if (m,protocol,s,ratio,cell) in B_LOOK:
                            body.append(' & '.join([LABELS[m],cell]+[pm(B_LOOK[m,protocol,s,ratio,cell],k) for k in METRICS])+r' \\')
    longtable('B_cells','tab:screened-B-cells','All frozen B cells and A-anchor cells. Six metrics are equal-fold means $\\pm$ sample SD (five folds). Only prespecified applicable method--cell combinations are present; no missing combination is inferred.',['Method','Cell']+METRICS,body,True)
    body=[]
    for protocol in ['strict','standard']:
        for s in ['warm','blind_both']:
            for ratio in ['1','10']:
                body.append(group_title(f'{protocol.capitalize()} | {SLABEL[s]} | outer 1:{ratio}',4))
                for m in METHODS:
                    for contrast in CONTRASTS:
                        rr=[z for z in CONTRAST if z['method']==m and z['protocol']==protocol and z['setting']==s and z['ratio']==ratio and z['contrast']==contrast]
                        if not rr:continue
                        d={z['metric']:z for z in rr}
                        body.append(' & '.join([LABELS[m],CLABEL[contrast]]+[f"{float(d[k]['mean_paired_difference']):+.3f} $\\pm$ {float(d[k]['SD_paired_difference']):.3f}" for k in ['AUROC','AP']])+r' \\')
    longtable('B_contrasts','tab:screened-B-contrasts','All 160 prespecified applicable paired contrast combinations: AUROC and AP differences (first cell minus reference, as frozen) $\\pm$ sample SD of five paired fold differences. Bars/SD are not confidence intervals. The complete TSV retains all 640 original rows, including MCC and F1.',['Method','Prespecified contrast','AUROC difference','AP difference'],body)
    body=[]
    for protocol in ['strict','standard']:
        for ratio in ['1','10']:
            body.append(group_title(f'{protocol.capitalize()} | source-record proxy | outer 1:{ratio}',7))
            for m in METHODS:body.append(' & '.join([LABELS[m]]+[pm(C_LOOK[m,protocol,ratio],k) for k in METRICS])+r' \\')
    longtable('C1_full','tab:screened-C1-full','All C1 metrics: equal means over five fold means within each of five draws, then equal means over draws. SD is the sample SD of the five draw means, not a 25-run SD. Source records are supplied provenance proxies, not established independent biological cohorts.',['Method']+METRICS,body,True)
    body=[]
    for protocol in ['strict','standard']:
        for ratio in ['1','10']:
            body.append(group_title(f'{protocol.capitalize()} | outer 1:{ratio}',5))
            for m in METHODS:
                for metric in ['AUROC','AP']:
                    r=CI_LOOK[m,protocol,ratio,metric];body.append(' & '.join([LABELS[m],metric,fnum(r['point']),fnum(r['lower']),fnum(r['upper'])])+r' \\')
    longtable('C1_intervals','tab:screened-C1-intervals','Original 95\\% percentile intervals from 2000 conditional source-record bootstrap replicates. Models, fixed negative pools and RNA universe are held fixed; intervals do not include training uncertainty or all endpoint dependencies. No replicate is regenerated.',['Method','Metric','Point','Lower','Upper'],body)

def verify_aggregates():
    checked=0
    for phase,summary in [('A',A),('B',B),('C1',C)]:
        raw=RAW[phase];groups=defaultdict(list)
        for r in raw:groups[r['method'],r['protocol'],r['setting'],r['cell'],r['ratio']].append(r)
        for r in summary:
            g=groups[r['method'],r['protocol'],r['setting'],r['cell'],r['ratio']]
            for metric in METRICS:
                if phase=='C1':
                    drawmeans=[]
                    for draw in range(5):
                        vals=[float(z[metric]) for z in g if int(z['draw'])==draw];ensure(len(vals)==5,'C1 five equal folds');drawmeans.append(statistics.mean(vals))
                    vals=drawmeans
                else:
                    vals=[float(z[metric]) for z in g];ensure(len(vals)==5,'Five-fold aggregate')
                close(statistics.mean(vals),r[metric+'_mean']);close(statistics.stdev(vals),r[metric+'_std']);checked+=2
    for r in CONTRAST:
        vals=[float(r[f'fold{i}_difference']) for i in range(5)]
        close(statistics.mean(vals),r['mean_paired_difference']);close(statistics.stdev(vals),r['SD_paired_difference']);checked+=2
    for r in CI:
        ensure(r['replicates']=='2000' and float(r['lower'])<=float(r['upper']),'Frozen bootstrap interval')
        close(r['point'],C_LOOK[r['method'],r['protocol'],r['ratio']][r['metric']+'_mean']);checked+=1
    CHECKS['independent_equal_weight_and_paired_arithmetic_comparisons']=checked

def style():
    font=font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    CHECKS['font_family']='Times New Roman';CHECKS['font_present']=bool(font)

def savefig(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',metadata={'Creator':'SEAGLMI benchmark'})
    fig.savefig(FIG/(name+'.png'),dpi=250,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',metadata={'Creator':'SEAGLMI benchmark'})
    plt.close(fig)

def headline(metric):
    fig,axs=plt.subplots(1,4,figsize=(7.35,4.55),sharey=True,layout='constrained')
    yy=np.arange(len(METHODS))[::-1]
    for ax,s in zip(axs,SETTINGS):
        for y,m in zip(yy,METHODS):
            strict=A_LOOK[m,'strict',s,'1'];standard=A_LOOK[m,'standard',s,'1'];a=float(strict[metric+'_mean']);b=float(standard[metric+'_mean'])
            for protocol,record in [('strict',strict),('standard',standard)]:
                mean=float(record[metric+'_mean']);sd=float(record[metric+'_std']);PLOT_ROWS.append(dict(figure='fig_screened_headline'+('' if metric=='AUROC' else '_AP'),phase='A',method=m,protocol=protocol,setting=s,ratio='1',metric=metric,contrast='',point=mean,lower=mean-sd,upper=mean+sd,uncertainty='five_fold_sample_SD'))
            ax.plot([a,b],[y,y],color='#B8A99E',lw=1,zorder=1)
            ax.errorbar(a,y+.08,xerr=float(strict[metric+'_std']),fmt='o',color=ORANGE,ms=4,capsize=2,lw=.8,zorder=3)
            ax.errorbar(b,y-.08,xerr=float(standard[metric+'_std']),fmt='s',color=BLUE,mfc='white',ms=4,capsize=2,lw=.8,zorder=2)
        ax.set_title(SLABEL[s].replace('Inherited warm split','Inherited warm\nsplit'),fontsize=8.5);ax.set_xlabel(metric);ax.grid(axis='x',color='#E8E2DC',lw=.5);ax.set_yticks(yy,LABELS.values());ax.set_ylim(-.8,13.8);ax.set_xlim(-.04,1.04);ax.set_xticks([0,.5,1])
    handles=[plt.Line2D([],[],marker='o',ls='',color=ORANGE,label='Strict'),plt.Line2D([],[],marker='s',ls='',color=BLUE,mfc='white',label='Standard')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,title='Five-fold means; bars = sample SD; outer 1:1.\nProtocol pipeline comparison, not a sampling-only effect.')
    savefig(fig,'fig_screened_headline'+('' if metric=='AUROC' else '_AP'))

def source_figure(ratio):
    fig,axs=plt.subplots(1,4,figsize=(7.35,4.55),sharey=True,layout='constrained');yy=np.arange(14)[::-1]
    for ax,(protocol,metric) in zip(axs,[(p,k) for p in ['strict','standard'] for k in ['AUROC','AP']]):
        for y,m in zip(yy,METHODS):
            r=CI_LOOK[m,protocol,ratio,metric];p,l,u=map(float,[r['point'],r['lower'],r['upper']]);PLOT_ROWS.append(dict(figure='fig_screened_source_CI'+('' if ratio=='1' else '_ratio10'),phase='C1',method=m,protocol=protocol,setting='source',ratio=ratio,metric=metric,contrast='',point=p,lower=l,upper=u,uncertainty='original_2000_conditional_source_record_bootstrap_95pct_CI'));color=ORANGE if protocol=='strict' else BLUE;ax.plot([l,u],[y,y],color=color,lw=1);ax.plot([l,l,u,u],[y-.06,y+.06,y-.06,y+.06],color=color,ls='',marker='|',ms=3);ax.plot(p,y,'o',ms=3.5,color=color)
        ax.set_title(protocol.capitalize()+' | '+metric,fontsize=8.5);ax.set_xlabel(metric);ax.set_yticks(yy,LABELS.values());ax.set_xlim(-.03,1.03);ax.set_xticks([0,.5,1]);ax.grid(axis='x',color='#E8E2DC',lw=.5)
    fig.supxlabel('Source-record proxy; original 2000 conditional bootstrap replicates.\n95% percentile intervals; outer 1:'+ratio+'.',fontsize=8)
    savefig(fig,'fig_screened_source_CI'+('' if ratio=='1' else '_ratio10'))

def forest(metric='AUROC',ratio='1',main=False):
    selected_contrasts=CONTRASTS[:4] if main else CONTRASTS
    fig,axs=plt.subplots(2,2 if main else 3,figsize=(7.35 if main else 9.0,5.8 if main else 11.2),layout='constrained');axs=np.ravel(axs)
    for ax,contrast in zip(axs,selected_contrasts):
        rr=[r for r in CONTRAST if r['metric']==metric and r['ratio']==ratio and r['contrast']==contrast and (not main or r['protocol']=='strict')]
        rr.sort(key=lambda r:(['strict','standard'].index(r['protocol']),['warm','blind_both'].index(r['setting']),METHODS.index(r['method'])))
        labels=[]
        for y,r in enumerate(rr[::-1]):
            mean=float(r['mean_paired_difference']);sd=float(r['SD_paired_difference']);PLOT_ROWS.append(dict(figure='fig_screened_B_contrasts_main' if main else f'fig_screened_B_contrasts_{metric}_ratio{ratio}',phase='B',method=r['method'],protocol=r['protocol'],setting=r['setting'],ratio=ratio,metric=metric,contrast=contrast,point=mean,lower=mean-sd,upper=mean+sd,uncertainty='five_paired_fold_difference_sample_SD'));ax.errorbar(mean,y,xerr=sd,fmt='o' if r['setting']=='warm' else 's',color=ORANGE if r['protocol']=='strict' else BLUE,ms=4,lw=1,capsize=2)
            labels.append(LABELS[r['method']]+' | '+('I-warm' if r['setting']=='warm' else 'both')+(' | '+r['protocol'] if not main else ''))
        ax.axvline(0,color='#93897E',lw=.8,ls='--');ax.set_yticks(range(len(labels)),labels,fontsize=7.5);ax.set_title(CLABEL[contrast]);ax.set_xlabel(metric+' paired difference');ax.grid(axis='x',color='#E8E2DC',lw=.5)
    fig.supxlabel('Paired five-fold means; bars = sample SD (not CI); no significance tests.\nI-warm = Inherited warm split; outer 1:'+ratio+'.',fontsize=8)
    name='fig_screened_B_contrasts_main' if main else f'fig_screened_B_contrasts_{metric}_ratio{ratio}'
    savefig(fig,name)

def main():
    global A,B,C,CONTRAST,CI,RAW,A_LOOK,B_LOOK,C_LOOK,CI_LOOK
    for folder in [TABLE,FIG]: folder.mkdir(parents=True,exist_ok=True)
    def load(name):
        with (DATA/name).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f,delimiter='\t'))
    A=load('A_summary.tsv');B=load('B_summary.tsv');C=load('C1_summary.tsv')
    CONTRAST=load('B_prespecified_paired_contrasts.tsv');CI=load('C1_source_bootstrap_intervals.tsv')
    RAW={'A':load('A_metrics_by_fold.tsv'),'B':load('B_metrics_with_A_anchor.tsv'),'C1':load('C1_metrics_by_fold.tsv')}
    ensure((len(A),len(B),len(C),len(CONTRAST),len(CI))==(224,144,56,640,112),'Complete reporting scope')
    A_LOOK={(r['method'],r['protocol'],r['setting'],r['ratio']):r for r in A}
    B_LOOK={(r['method'],r['protocol'],r['setting'],r['ratio'],r['cell']):r for r in B}
    C_LOOK={(r['method'],r['protocol'],r['ratio']):r for r in C}
    CI_LOOK={(r['method'],r['protocol'],r['ratio'],r['metric']):r for r in CI}
    verify_aggregates()
    for protocol in ['strict','standard']: main_table('A',protocol)
    main_table('C1');supplementary_tables();style()
    headline('AUROC');headline('AP');source_figure('1');source_figure('10');forest(main=True)
    for metric in ['AUROC','AP']:
        for ratio in ['1','10']:forest(metric,ratio)
    with (PAPER/'generated/replayed_plot_coordinates.tsv').open('w',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(k for r in PLOT_ROWS for k in r))
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(PLOT_ROWS)
    (PAPER/'generated/reporting_replay.json').write_text(json.dumps({'status':'frozen_reporting_replay_passed','checks':CHECKS,'points':len(PLOT_ROWS),'scope':'Aggregate/reporting replay only; not a second raw-prediction or bootstrap audit.'},indent=2)+'\n')
if __name__=='__main__':main()
