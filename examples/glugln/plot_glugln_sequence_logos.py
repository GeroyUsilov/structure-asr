"""Matched AA/3Di probability logos at random central alignment sites."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

BASE = Path(__file__).resolve().parent
SOURCE = Path('/Users/christopher/Desktop/aleks_claude/runpod_glugln_aa168_3di16_uncertainty/inputs')
OUT = BASE / 'glugln_sequence_logos'
FONT = '/System/Library/Fonts/Supplemental/Courier New Bold.ttf'
SEED = 42


def read(name, node):
    with (SOURCE / name).open() as h:
        reader=csv.DictReader(h)
        columns=[c for c in reader.fieldnames if c.startswith('p_')]
        rows=sorted(reader,key=lambda r:int(r['Site']))
    assert len(rows)==307 and all(r['Node']==node for r in rows)
    assert [int(r['Site']) for r in rows]==list(range(1,308))
    p=np.array([[float(r[c]) for c in columns] for r in rows])
    assert (p>=0).all() and (p.sum(1)>0).all()
    p/=p.sum(1,keepdims=True)
    return [c[2:] for c in columns],p


def logo(ax, alphabet, p, sites, color, label, lowercase=False):
    font=FontProperties(fname=FONT)
    paths={s:TextPath((0,0),s.lower() if lowercase else s,size=1,prop=font) for s in alphabet}
    # A common width scale retains the monospace glyph proportions in x.
    common_width=max(path.get_extents().width for path in paths.values())
    xscale=.78/common_width
    for x,site in enumerate(sites):
        bottom=0.
        for i in np.argsort(p[site-1],kind='stable'):
            height=p[site-1,i]
            if height<=0:continue
            path=paths[alphabet[i]]; box=path.get_extents()
            transform=(Affine2D().translate(-box.x0,-box.y0)
                       .scale(xscale,height/box.height)
                       .translate(x-box.width*xscale/2,bottom))
            patch=PathPatch(path,transform=transform+ax.transData,
                            facecolor=color,edgecolor='none',linewidth=0)
            ax.add_patch(patch)
            bottom+=height
        assert abs(bottom-1)<1e-12
    ax.set(xlim=(-.65,9.65),ylim=(0,1.03),ylabel='Probability')
    ax.set_yticks([0,.5,1])
    ax.set_xticks(range(10),[str(s) for s in sites])
    ax.text(0,1.10,label,transform=ax.transAxes,fontsize=12,color=color)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',length=0,pad=6)
    ax.grid(False)


def main():
    aa_alphabet,aa=read('AAseq_Node168.csv','Node168')
    di_alphabet,di=read('3Di_Node16.csv','Node16')
    # Central third, rounded to alignment positions 103..204 inclusive.
    logs=np.zeros_like(di)
    np.log2(di,out=logs,where=di>0)
    entropy=-(di*logs).sum(axis=1)
    starts=np.arange(103,196)
    window_entropy=np.array([entropy[s-1:s+9].mean() for s in starts])
    start=int(starts[np.argmax(window_entropy)])
    sites=np.arange(start,start+10)
    plt.rcParams.update({'font.family':'Gill Sans',
        'font.size':12,'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/GillSans.ttc')
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Courier New.ttf')
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
    fig.subplots_adjust(left=.115,right=.975,top=.9,bottom=.19,hspace=.52)
    logo(axes[0],aa_alphabet,aa,sites,'#2463A6','AA model')
    logo(axes[1],di_alphabet,di,sites,'#D97916','3Di model',lowercase=True)
    axes[0].tick_params(labelbottom=True)
    axes[1].set_xlabel('Alignment position',labelpad=9)
    fig.text(.115,.035,'Highest mean 3Di entropy among 10-site windows in the middle third.',fontsize=12)
    OUT.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'glugln_aa_3di_probability_logos.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    metadata={'sites':sites.tolist(),'search_range_inclusive':[103,204],
              'selection':'maximum mean direct 3Di entropy; 10 consecutive sites; earliest start breaks ties',
              'mean_3di_entropy_bits':float(entropy[sites-1].mean()),
              'aa_node':'Node168','3di_node':'Node16','height':'normalized posterior probability',
              'label_font':'Gill Sans','label_size_pt':12,'3di_display_case':'lowercase',
              'glyph_font':FONT,'aa_probabilities':aa[sites-1].tolist(),
              'aa_alphabet':aa_alphabet,'3di_probabilities':di[sites-1].tolist(),'3di_alphabet':di_alphabet}
    (OUT/'logo_data.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Positions:',', '.join(map(str,sites)))
    print('Mean 3Di entropy:',float(entropy[sites-1].mean()),'bits; previous region:',float(entropy[110:120].mean()))
    print(OUT)


if __name__=='__main__':main()
