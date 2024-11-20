#!/bin/bash

# Loop para cada pasta de 1_fold a 5_fold
for i in {1..5}; do
    # Definindo o nome da pasta de destino
    folder="${i}_fold"
    
    # Movendo e renomeando os arquivos para a pasta correta
    mv "${i}_foldconf_matrix.png" "${folder}/conf_matrix.png"
    mv "${i}_foldfold_n_${i}_2dconv_fixed.h5" "${folder}/fold_n_${i}_2dconv_fixed.h5"
    mv "${i}_foldloss_and_val_accuracy.png" "${folder}/loss_and_val_accuracy.png"
    mv "${i}_foldloss_and_val_loss.png" "${folder}/loss_and_val_loss.png"
    mv "${i}_foldplot_roc.png" "${folder}/plot_roc.png"
    
    echo "Arquivos movidos e renomeados para ${folder}"
done

echo "Organização concluída!"
