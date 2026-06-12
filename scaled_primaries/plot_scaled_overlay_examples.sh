#!/bin/bash
# Example usage of plot_scaled_overlay.py


# With signal (CE) component using physics-motivated yields
python plot_scaled_overlay.py \
    --variable "recomom_ttfront" \
    --dio ../file_lists/DIOtail95_MDC2025an_best_nomix.txt \
    --cosmic ../file_lists/Cosimcs_MDC2025an_nomix.txt \
    --ce ../file_lists/CeMLL_MDC2025an_best_nomix.txt \
    --rpc-ext ../file_lists/ExtRPC_MDC2025an_nomix.txt \
    --rpc-int ../file_lists/IntRPC_MDC2025an_nomix.txt \
    --data ../file_lists/MDS3c_1e-13_2.txt  \
    --output plots/recomom_ttfront_with_signal.pdf \
    --range 102 106 \
    --bins 80 \
    --title "" \
    --dio-yield 6017 \
    --cosmic-yield 520 \
    --rpc-ext-yield 1 \
    --rpc-int-yield 1 \
    --ce-yield 82 \
    --jobs 16 \
    --cut-lo 103.34 \
    --cut-hi 104.74 \
    --verbosity 2