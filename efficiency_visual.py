"""
Creates multiple plots for (1)each process ID and (2)a final plot of overlays of all processes. Each csv file with information of the cut efficiency can be added as user input; this code will generate a plot for each file and then a final plot of all the information overlayed. 

Usage:
    python efficiency_visual.py --cut_table eminus_DIO_cut_stats.csv 

Example:
    python plot_arxiv_global.py --cut_table /path/to/files/*
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
x_axis_labels = {
	"No cuts"          : "Tracker Acceptance",
	"is_reco_electron"      : "Reconstructed as Electron",
	"has_downstream"         : "Downstream selection",
	"upstream"  : "Upstream Selection",
	"has_trk_front_seg"      : "Crosses Tracker Front",
	"good_trkpid"      : "PID selection",
	"good_trkqual"      : "Track Quality",
	"within_t0err"      : "Time error",
	"has_hits"      : "Has Hits In Tracker",
	"has_st"      : "From Stopping Target",
	"no_opa"      : "OPA veto",
	"no_crv_quality"      : "CRV Quality Cut",
	"no_crv_timewindow"      : "CRV Time Cut",
	"no_crv_veto"      : "CRV Veto",
	"pz_over_pt"      : "pz/pt",
	"good_trigger"      : "Trigger Cut"
	#"final mom selection"      : "Momentum Cut",
	#"final time selection"      : "Time Cut"
}
def plot_cut_efficiency(cut_selec, events, frac,my_title):
    fig=plt.figure(figsize=(12, 6))
    plt.title(my_title, fontsize=16, pad=20)
    labels = [x_axis_labels.get(cut,cut) for cut in cut_selec]
    plt.bar(labels, events)

    
    offset = max(events) * 0.01 
    
    for i in range(len(cut_selec)):
    	val = float(frac[i])
    	if 0 < val < 0.01:
    		label_text = f"{val:.1e}%"	
    	else:
    		label_text = f"{frac[i]:.2f}%"
    	
    	plt.text(labels[i], events[i] + offset, label_text, ha='center', va='bottom')
          
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout() 
    plt.show()

    plt.close(fig)

def compare_cuts(ax, cut_selec, events, frac, label_name, fill_alpha=0.2):
    frac_num = np.array(frac,dtype=float)
    x_indices = np.arange(len(frac_num))
    
    line, = ax.plot(x_indices, frac_num, linewidth=2, label=label_name)
    current_color = line.get_color()
    ax.fill_between(x_indices, frac_num, 0, alpha=fill_alpha,color=current_color)
    
    labels = [x_axis_labels.get(cut, cut) for cut in cut_selec]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    last_index = len(cut_selec) - 1
    val_frac = frac_num[last_index]
    if 0 < val_frac < 0.01:
    	pct_val = f"{val_frac:.1e}%"	
    else:
    	pct_val = f"{val_frac:.2f}%"
    line.set_label(f"{label_name} ({pct_val})")
    return line
def main():
    parser = argparse.ArgumentParser(
        description="Generate Efficiency Visualisation Plots", 
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--cut_table", type=str, nargs='+', required=True, help="path to one or more cut flow tables")
    
    args = parser.parse_args()
    print(f"Ready to process data from: {args.cut_table}")
    compare_datasets = []
 

    for file_name in args.cut_table:
        print(f"\nProcessing data from: {file_name}")
        df = pd.read_csv(file_name)
        
        '''if "IPADIO" in file_name.upper():
        	my_title = "DIO from IPA Cut Flow"
        	short_label = "IPA DIO"'''
        if "CE" in file_name.upper():
        	my_title = "Conversion Electron Cut Flow"
        	short_label = "CE"
        elif "DIO" in file_name.upper():
        	my_title = "Decay in Orbit Cut Flow"
        	short_label = "DIO"
        elif "EXTRPC" in file_name.upper():
        	my_title = " External Radiative Pion Capture Cut Flow"
        	short_label=" External RPC"
        elif "INTRPC" in file_name.upper():
        	my_title = "Internal Radiative Pion Capture Cut Flow"
        	short_label="Internal RPC"
        elif "COSMICS" in file_name.upper():
        	my_title = "Cosmic Ray Cut Flow"
        	short_label = "Cosmics"
        else:
        	my_title = f"Cut Flow for {file_name}"
        	short_label = file_name
        '''elif "EXTRMC" in file_name.upper():
        	my_title = "External Radiative Muon Capture Cut Flow"
        	short_label = "eRMC"
        elif "INTRMC" in file_name.upper():
        	my_title = " Internal Radiative Muon Capture Cut Flow"
        	short_label = "Internal RMC"'''

        plot_cut_efficiency(df["Cuts"], df["Events"], df["Absolute_w_eff[%]"],my_title)
        
        compare_datasets.append({"cut":df["Cuts"],"events":df["Events"],"frac":df["Absolute_w_eff[%]"],"label":short_label})
        
    print("\nGenerating final comparison overlay plot")
    
    
    fig,ax=plt.subplots(figsize=(12,7))
    ax.set_title("Cut Flow For All Processes",fontsize=20,pad=20,fontweight="bold")
    ax.set_ylabel("Absolute Efficiency [%]",fontsize=14,fontweight="bold")
    ax.set_xlabel("Cut Selection",fontsize=14,fontweight="bold")    

    for data in compare_datasets:
    	compare_cuts(ax,data["cut"],data["events"],data["frac"],data["label"],fill_alpha=0.2)
    ax.tick_params(axis = "x",rotation=45, labelsize=14)
    ax.tick_params(axis = "y", labelsize=14)
    ax.legend(loc="upper right",fontsize=12)
    num_cuts = len(compare_datasets[0]["cut"])
    ax.set_xlim(-0.5,num_cuts-0.2)
    plt.tight_layout()
    print("\nDisplaying final comparison overlay plot")
    plt.show()
    plt.close(fig)
    
    
    fig,ax=plt.subplots(figsize=(12,7))
    #ax.set_title("Cut Flow Log Plot",fontsize=20,pad=20)
    ax.set_ylabel("Absolute Efficiency [%]",fontsize=16)
    ax.set_xlabel("Cut Selection",fontsize=16)

    line_data = []
    for data in compare_datasets:
        line = compare_cuts(ax, data["cut"], data["events"], data["frac"], data["label"], fill_alpha=0)
        line.set_label(data["label"])
        frac_num = np.array(data["frac"], dtype=float)
        line_data.append({'line': line, 'frac_num': frac_num})

    ax.tick_params(axis="x", rotation=45, labelsize=14)
    ax.tick_params(axis = "y", labelsize=14)
    ax.legend(loc="lower left", fontsize=12)
    num_cuts = len(compare_datasets[0]["cut"])
    last_x = num_cuts - 1

    # Collect endpoint annotation data; fall back to last non-zero cut for zero-efficiency processes
    endpoint_data = []
    for ld in line_data:
        frac_num = ld['frac_num']
        nonzero_idx = np.where(frac_num > 0)[0]
        if len(nonzero_idx) == 0:
            continue
        ann_x = int(nonzero_idx[-1])
        ann_y = float(frac_num[ann_x])
        val_frac = float(frac_num[-1])
        if val_frac <= 0:
            continue
        pct_val = f"{val_frac:.1e}%" if val_frac < 0.01 else f"{val_frac:.2f}%"
        endpoint_data.append({'ann_x': ann_x, 'ann_y': ann_y, 'sort_y': val_frac,
                               'pct': pct_val, 'color': ld['line'].get_color()})

    # Stagger labels per x-group in log space, placing each label just right of its line's endpoint
    min_log_gap = 0.35
    for group_x in sorted(set(d['ann_x'] for d in endpoint_data)):
        group = [d for d in endpoint_data if d['ann_x'] == group_x]
        group.sort(key=lambda d: d['sort_y'])
        log_ys_g = [np.log10(d['sort_y']) for d in group]
        nudged = list(log_ys_g)
        for i in range(1, len(nudged)):
            if nudged[i] - nudged[i - 1] < min_log_gap:
                nudged[i] = nudged[i - 1] + min_log_gap
        for i, d in enumerate(group):
            d['text_y'] = 10 ** nudged[i]

    text_offset = 0.4
    for d in endpoint_data:
        text_x = d['ann_x'] + text_offset
        text_y = d['text_y']
        needs_arrow = abs(np.log10(text_y) - np.log10(d['ann_y'])) > 0.1
        ax.annotate(
            d['pct'],
            xy=(d['ann_x'], d['ann_y']),
            xytext=(text_x, text_y),
            color=d['color'], fontsize=12, fontweight='bold', ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=d['color'], alpha=0.0, lw=1.2) if needs_arrow else None
        )

    max_ann_x = max(d['ann_x'] for d in endpoint_data) if endpoint_data else last_x
    ax.set_xlim(-0.5, max_ann_x + 2.5)
    ax.set_yscale('log')
    plt.tight_layout()
    print("\nDisplaying final comparison overlay plot")
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
