"""
Regenerates Figs. 1-4 of the manuscript from the released result files.

Usage:  python scripts/make_figures.py --data data/ --out figures/
Requires: numpy, pandas, scipy, matplotlib, opfunu==1.0.1
"""
import argparse, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import rankdata, wilcoxon

plt.rcParams.update({'font.family':'serif','font.size':9,'axes.linewidth':0.7,
                     'figure.dpi':300,'savefig.bbox':'tight'})

def fig1_ranks(data, out):
    """Fig. 1 - mean Friedman rank, official CEC 2017, 30D and 50D."""
    full = pd.read_csv(os.path.join(data,'results_full.csv'))
    d={}
    for dim in (30,50):
        piv=full[full.dim==dim].groupby(['fid','algorithm']).fbest.mean().unstack()
        R=np.apply_along_axis(rankdata,1,piv.values)
        mfr=pd.Series(R.mean(axis=0),index=piv.columns)
        q=piv['QSO-time']; rows=[]
        for c in piv.columns:
            if c=='QSO-time': continue
            rows.append([c, wilcoxon(q,piv[c])[1]])
        rows.sort(key=lambda r:r[1]); m=len(rows)
        sig={r[0]:(min(1,r[1]*(m-i))<0.05) for i,r in enumerate(rows)}
        d[dim]=(mfr,sig)
    order=d[50][0].sort_values(ascending=False).index.tolist()
    fig,ax=plt.subplots(figsize=(6.6,4.2)); y=np.arange(len(order))
    for dim,mk,cl,off in [(30,'o','#12406e',-0.16),(50,'s','#c0392b',0.16)]:
        mfr,sig=d[dim]
        for i,a in enumerate(order):
            filled = a=='QSO-time' or sig.get(a,False)
            ax.scatter(mfr[a], y[i]+off, marker=mk, s=34,
                       facecolor=cl if filled else 'white', edgecolor=cl,
                       linewidth=0.9, zorder=3, label=f'{dim}D' if i==0 else None)
        ax.plot([mfr[a] for a in order], y+off, color=cl, lw=0.5, alpha=0.35, zorder=1)
    ax.axvline(d[50][0]['QSO-time'],color='#c0392b',ls=':',lw=0.9,alpha=0.7,zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(['QSO' if a=='QSO-time' else a for a in order],fontsize=8)
    ax.set_xlabel('Mean Friedman rank (lower is better)',fontsize=8.5)
    ax.grid(axis='x',alpha=0.25,lw=0.4); ax.legend(fontsize=8,frameon=False,loc='lower right')
    fig.savefig(os.path.join(out,'fig1_ranks.pdf')); plt.close(fig)

def fig3_switching(data, out):
    """Fig. 3 - autoinducer concentration against adaptive threshold, 50D seed 42."""
    tr=json.load(open(os.path.join(data,'switching_trajectories_50D_seed42.json')))
    panels=[('1','F1 · Unimodal'),('5','F5 · Multimodal'),
            ('15','F15 · Hybrid'),('23','F23 · Composition')]
    fig=plt.figure(figsize=(7.2,5.2))
    gs=GridSpec(4,2,figure=fig,height_ratios=[1,0.16,1,0.16],hspace=0.12,wspace=0.18)
    for k,(fid,title) in enumerate(panels):
        row=(k//2)*2; col=k%2
        axm=fig.add_subplot(gs[row,col]); axp=fig.add_subplot(gs[row+1,col],sharex=axm)
        C=np.array(tr[fid]['quorum']); th=np.array(tr[fid]['theta']); ph=np.array(tr[fid]['phase'])
        t=np.arange(len(C))
        axm.fill_between(t,0,1,where=ph.astype(bool),color='#dbe9f6',lw=0,
                         transform=axm.get_xaxis_transform(),zorder=0)
        axm.plot(t,C,color='#12406e',lw=0.65,zorder=3,label=r'$C^t$')
        axm.plot(t,th,color='#c0392b',lw=1.2,ls='--',zorder=4,label=r'$\theta^t$')
        axm.set_ylim(0,1.02); axm.set_xlim(0,len(C)-1); axm.set_title(title,fontsize=8.6,pad=3)
        axm.tick_params(labelbottom=False,labelsize=7.5)
        axm.text(0.985,0.05,f'{int((np.diff(ph)!=0).sum())} transitions · {100*ph.mean():.0f}% collective',
                 transform=axm.transAxes,ha='right',va='bottom',fontsize=7,
                 bbox=dict(fc='white',ec='0.7',lw=0.4,pad=2))
        axp.imshow(ph.reshape(1,-1),aspect='auto',
                   cmap=plt.cm.colors.ListedColormap(['#f4f4f4','#12406e']),
                   extent=[0,len(C)-1,0,1],interpolation='nearest',vmin=0,vmax=1)
        axp.set_yticks([]); axp.tick_params(labelsize=7.5)
        if row==2: axp.set_xlabel('Iteration',fontsize=8)
        else: axp.tick_params(labelbottom=False)
        if col==0: axm.set_ylabel('Signal / threshold',fontsize=8)
    h,l=fig.axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper right',bbox_to_anchor=(0.995,1.005),ncol=2,fontsize=8,frameon=False)
    fig.savefig(os.path.join(out,'fig3_switching.pdf')); plt.close(fig)

def fig4_budget(data, out):
    """Fig. 4 - rank stability across evaluation budgets."""
    d=pd.read_csv(os.path.join(data,'results_budget_ladder.csv'))
    d=d[~((d.dim==30)&(d.max_iter==10000))]
    show=['QSO-time','DE','GWO','HHO','MPA','PSO']
    colr={'QSO-time':'#c0392b','DE':'#12406e','GWO':'#6a3d9a',
          'HHO':'#777777','MPA':'#8e6c1f','PSO':'#2e7d32'}
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True)
    for ax,dim in zip(axes,[30,50]):
        fes=[]; series={a:[] for a in show}
        for mi in sorted(d[d.dim==dim].max_iter.unique()):
            sub=d[(d.dim==dim)&(d.max_iter==mi)]
            piv=sub.groupby(['fid','algorithm']).fbest.mean().unstack().dropna(axis=1)
            mfr=pd.Series(np.apply_along_axis(rankdata,1,piv.values).mean(axis=0),index=piv.columns)
            fes.append(int(sub.fes.iloc[0]))
            for a in show: series[a].append(mfr[a])
        for a in show:
            ax.plot(fes,series[a],marker='o',ms=4,lw=1.6 if a=='QSO-time' else 0.9,
                    color=colr[a],label='QSO' if a=='QSO-time' else a)
        ax.set_xscale('log'); ax.set_title(f'{dim}D',fontsize=9)
        ax.set_xlabel('Function evaluations',fontsize=8.5)
        ax.grid(alpha=0.25,lw=0.4); ax.invert_yaxis()
    axes[0].set_ylabel('Mean Friedman rank',fontsize=8.5)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',bbox_to_anchor=(0.5,1.10),ncol=6,fontsize=8,frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out,'fig4_budget.pdf')); plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data'); ap.add_argument('--out',default='figures')
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    fig1_ranks(a.data,a.out); print('Fig. 1 written')
    fig3_switching(a.data,a.out); print('Fig. 3 written')
    fig4_budget(a.data,a.out); print('Fig. 4 written')
    print('Fig. 2 (convergence) requires opfunu and qso.py — see scripts/make_fig2.py')
