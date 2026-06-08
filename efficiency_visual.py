import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def plot_cut_efficiency(cut_selec, events, frac,my_title):

    fig=plt.figure(figsize=(12, 6))
    plt.title(my_title, fontsize=16, pad=20)
    plt.bar(cut_selec, events)
    
    offset = max(events) * 0.01 
    
    for i in range(len(cut_selec)):
        label_text = f"{frac[i]:.1f}%"
        plt.text(cut_selec[i], events[i] + offset, label_text, ha='center', va='bottom')
        
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout() 
    plt.show()

    plt.close(fig)


_last_y_data = None

def compare_cuts(ax, cut_selec, events, frac, label_name):
    global _last_y_data
    
    x_indices = np.arange(len(events))
    
    line, = ax.plot(x_indices, events, linewidth=2, label=label_name)
    current_color = line.get_color()

    if _last_y_data is not None:
        ax.fill_between(x_indices, events, _last_y_data, alpha=0.25, color=current_color)
    else:
        ax.fill_between(x_indices, events, 0, alpha=0.25, color=current_color)
        

    _last_y_data = events.values if hasattr(events, 'values') else np.array(events)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(cut_selec, rotation=45, ha='right')
    
    last_index = len(cut_selec) - 1
    val_frac = frac.values[last_index] if hasattr(frac, 'values') else frac[last_index]
    val_events = events.values[last_index] if hasattr(events, 'values') else events[last_index]
    
    label_text = f"{val_frac:.1f}%"
    ax.annotate(
        label_text,
        xy=(last_index, val_events), 
        xytext=(6, 0),                       
        textcoords="offset points",          
        ha='left',                           
        va='center',                         
        fontsize=10,
        fontweight='bold',
        color=current_color                  
    )

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
        
        if "DIO" in file_name.upper():
        	my_title = "Decay in Orbit Cut Flow"
        	short_label = "DIO"
        elif "CE" in file_name.upper():
        	my_title = "Conversion Electron Cut Flow"
        	short_label = "CE"
        elif "RPC" in file_name.upper():
        	my_title = "Radiative Pion Capture Cut Flow"
        	short_label="RPC"
        elif "RMC" in file_name.upper():
        	my_title = "Radiative Muon Capture Cut Flow"
        	short_label = "RMC"
        elif "Cosmics" in file_name.upper():
        	my_title = "Cosmic Ray Cut Flow"
        	short_label = "Cosmics"
        else:
        	my_title = f"Cut Flow for {file_name}"
        	short_label = file_name
        plot_cut_efficiency(df["Cut"], df["Events Passing"], df["Absolute [%]"],my_title)
        compare_datasets.append({"cut":df["Cut"],"events":df["Events Passing"],"frac":df["Absolute [%]"],"label":short_label})
    print("\nGenerating final comparison overlay plot")
    fig,ax=plt.subplots(figsize=(12,6))
    ax.set_title("Process ID Cut Flow Comparison",fontsize=16,pad=20)
    ax.set_ylabel("Events Passing")
    ax.set_xlabel("Cut Selection") 
    
    for data in compare_datasets:
    	compare_cuts(ax,data["cut"],data["events"],data["frac"],data["label"])
    ax.tick_params(axis = "x",rotation=45)
    ax.legend(loc="upper right",fontsize=12)
    num_cuts = len(compare_datasets[0]["cut"])
    ax.set_xlim(-0.5,num_cuts-0.2)
    plt.tight_layout()
    print("\nDisplaying final comparison overlay plot")
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
