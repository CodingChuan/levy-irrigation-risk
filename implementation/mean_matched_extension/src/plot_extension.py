"""Plot all registered mechanism rows and complete saved-policy action sets."""
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
from extension import ROOT,OLD

OUT=ROOT/'figures'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,'axes.titlesize':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
COLORS=['#626D77','#B56A30','#007C91']
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/(name+'.alignment.json'),strict=True)
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(OUT/(name+'.tiff'),dpi=600,facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=300,facecolor='white')
    plt.close(fig)

def mechanisms():
    old=json.loads((OLD/'matrix_scalar.json').read_text())['rows'];new=json.loads((ROOT/'runs/matrix.json').read_text())['rows']
    groups=[[x for x in old if x['intensity']==0],new,[x for x in old if x['intensity']==.12]]
    names=['Brownian','Mean-matched diffusion','Jump diffusion']
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={'left':.25,'right':.97,'bottom':.22,'top':.84,'wspace':.30})
    for k,(ax,field,title,xmax) in enumerate(zip(axes,['initial_risk_no_cost','mixture_cost'],['No-irrigation failure risk','Water at risk limit 0.20'],[.7,.21])):
        for i,rows in enumerate(groups):
            fine=next(x for x in rows if x['h']==.0025 and x['upper']==1.55)
            v=fine[field];lo=min(x[field] for x in rows);hi=max(x[field] for x in rows)
            ax.plot([lo,hi],[2-i]*2,color=COLORS[i],lw=2)
            ax.scatter([v],[2-i],s=28,color=COLORS[i],zorder=3)
            ax.annotate(f'{v:.6f}',(v,2-i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
        ax.set(xlim=(0,xmax),ylim=(-.5,2.6),yticks=[2,1,0],yticklabels=names if k==0 else ['', '', ''],title=title,xlabel='Failure probability' if k==0 else 'Expected water (model units)')
        ax.text(-.12,1.08,chr(97+k),transform=ax.transAxes,fontweight='bold')
    save(fig,'mean_matched_comparison')
    rows=[]
    for name,group in zip(names,groups):
        for row in group:rows.append({'mechanism':name,**{k:row[k] for k in ['upper','h','initial_risk_no_cost','initial_risk_min','mixture_cost','mixture_risk','duality_gap']}})
    with (OUT/'mechanism_source.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def policies():
    rows=json.loads((ROOT/'runs/policy_readout.json').read_text())['rows']
    cm=ListedColormap(['#FFFFFF','#007C91','#D68D32','#82669D']);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    cats=[];grids=[]
    for row in rows:
        ar=[]
        for typ in ['risky','safe']:
            with np.load(ROOT/'runs'/row['endpoints'][row[typ+'_label']]['actions_file']) as z:ar.append(z['actions']>0);g=z['grid']
        cats.append(np.where(ar[0]&ar[1],1,np.where(ar[0],2,np.where(ar[1],3,0))))
        grids.append(g)
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.5),gridspec_kw={'left':.085,'right':.98,'bottom':.37,'top':.84,'wspace':.65})
    for i,ax in enumerate(axes[:2]):
        g=grids[i];ax.imshow(cats[i][:,4,:].T,origin='lower',extent=[-.5,19.5,g[0]-.00125,g[-1]+.00125],aspect='auto',cmap=cm,norm=norm,interpolation='none')
        ax.set(xlabel='Decision day',ylabel='Soil moisture',title='Brownian' if i==0 else 'Jump diffusion',xticks=[0,5,10,15,19],ylim=(.35,1.55))
    ax=axes[2]
    for i,row in enumerate(rows):ax.plot(np.arange(20),row['mixture_daily_water'],color=COLORS[0 if i==0 else 2],lw=1.6,label='Brownian' if i==0 else 'Jump diffusion')
    ax.set(xlabel='Decision day',ylabel='Water per day',title='Water allocation',xlim=(0,19),xticks=[0,5,10,15,19]);ax.ticklabel_format(axis='y',style='plain');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.32),frameon=False)
    for i,ax in enumerate(axes):ax.text(-.18,1.08,chr(97+i),transform=ax.transAxes,fontweight='bold')
    handles=[Patch(facecolor=cm(i),edgecolor='#aaa',label=s) for i,s in enumerate(['Neither policy','Both policies','Riskier endpoint only','Safer endpoint only'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.34,.02),ncol=2,fontsize=8,frameon=False)
    save(fig,'irrigation_decisions')
    fig,axes=plt.subplots(2,4,figsize=(7.2,4.8),gridspec_kw={'left':.14,'right':.98,'bottom':.20,'top':.90,'wspace':.35,'hspace':.45})
    for i,row in enumerate(rows):
        for j,b in enumerate([1,2,3,4]):
            ax=axes[i,j];g=grids[i];ax.imshow(cats[i][:,b,:].T,origin='lower',extent=[-.5,19.5,g[0]-.00125,g[-1]+.00125],aspect='auto',cmap=cm,norm=norm,interpolation='none')
            ax.set(xticks=[0,10,19],ylim=(.35,1.55),title=f'Budget {model_budgets[b]:.2f}',xlabel='Decision day' if i==1 else '',ylabel=('Brownian\nSoil moisture' if i==0 else 'Jump diffusion\nSoil moisture') if j==0 else '')
            ax.text(-.18,1.08,chr(97+i*4+j),transform=ax.transAxes,fontweight='bold')
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,fontsize=9,frameon=False)
    save(fig,'all_budget_decisions')
    with (OUT/'daily_water_source.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['mechanism','day','expected_water'])
        for i,row in enumerate(rows):
            for day,v in enumerate(row['mixture_daily_water']):w.writerow(['Brownian' if i==0 else 'Jump diffusion',day,v])

if __name__=='__main__':
    model_budgets=[0,.06,.14,.22,.30]
    mechanisms();policies()
