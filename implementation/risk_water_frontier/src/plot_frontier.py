"""Plot the complete registered risk-limit grid; do not interpolate optima."""
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})

def main():
    assert json.loads((ROOT/'runs/completion.json').read_text())['status']=='PASS'
    data=json.loads((ROOT/'runs/frontier.json').read_text());assert data['complete'] and len(data['rows'])==42
    fig,ax=plt.subplots(figsize=(6.2,3.5));fig.subplots_adjust(left=.14,right=.97,bottom=.20,top=.93)
    for name,col,marker in [('Brownian','#626D77','o'),('Mean-matched diffusion','#B56A30','s'),('Jump diffusion','#007C91','D')]:
        rows=[r for r in data['rows'] if r['mechanism']==name];x=np.array([r['risk_limit'] for r in rows]);y=np.array([r['expected_water'] if r['status']=='FEASIBLE' else np.nan for r in rows]);assert len(rows)==14 and np.all(np.diff(x)>0)
        ax.plot(x,y,color=col,marker=marker,markersize=3.5,lw=1.5,label=name)
        k=np.flatnonzero(x==.2)[0];ax.scatter([x[k]],[y[k]],s=55,facecolors='white',edgecolors=col,linewidths=1.5,zorder=4)
    ax.axvline(.2,color='#8C9499',lw=.8,ls='--',zorder=0)
    ax.text(.208,.188,'Risk limit 0.20',fontsize=9,va='top',color='#50585D')
    ax.set(xlim=(0,.66),ylim=(-.004,.20),xlabel='Allowed failure probability',ylabel='Minimum expected water (model units)',xticks=np.arange(0,.7,.1),yticks=np.arange(0,.21,.05))
    ax.legend(loc='upper right',fontsize=9)
    fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=ROOT/'figures/risk_water_curve.alignment.json',strict=True)
    fig.savefig(ROOT/'figures/risk_water_curve.pdf',facecolor='white');fig.savefig(ROOT/'figures/risk_water_curve.svg',facecolor='white');fig.savefig(ROOT/'figures/risk_water_curve.tiff',dpi=600,facecolor='white');fig.savefig(ROOT/'figures/risk_water_curve.png',dpi=300,facecolor='white');plt.close(fig)
    with (ROOT/'figures/risk_water_source.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['mechanism','risk_limit','status','expected_water','achieved_risk','source']);w.writeheader()
        for r in data['rows']:w.writerow({'mechanism':r['mechanism'],'risk_limit':r['risk_limit'],'status':r['status'],'expected_water':r['expected_water'],'achieved_risk':r['risk'],'source':r['source']})

if __name__=='__main__':main()
