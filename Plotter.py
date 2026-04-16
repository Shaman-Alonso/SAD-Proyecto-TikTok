import matplotlib.pyplot as plt
import seaborn as sn

class Plotter:
    def __init__(self, args):
        self.args = args

    def __plot_metricas(self, rdo_df, cm, cr):
        args = self.args
        # Hemos experimentado con estas librerías para mostrar las métricas y demases de manera más visual

        # Creamos el lienzo (ancho x alto)
        fig = plt.figure(figsize=(12, 25))

        # === Matriz de confusión ===
        # Creamos una caja de dos filas para representar los gráficos (fila x col x índice)
        ax1 = fig.add_subplot(3, 1, 1)  # ax1 es la primera fila, donde irá la matriz
        sn.heatmap(cm, annot=True, cmap="Greens",
                   ax=ax1)  # Que aparezcan numeros y colorinchis verdes (annot es para que se vea la frec abs)

        # Renombrar filas y columnas
        ax1.set_xlabel('Predicción', fontsize=12)
        ax1.set_ylabel('Real', fontsize=12)
        ax1.set_title('Matriz de Confusión', fontsize=14, pad=20)

        # === Métricas ===
        ax_txt = fig.add_subplot(3, 1, 2)
        ax_txt.axis('off')
        ax_txt.text(0.5, 0.5, f"Informe de clasificación:\n\n{cr}", fontsize=10, va='center', ha='center',
                    family='monospace')

        # Cogemos el top 10
        metrica_col = f"mean_test_{args.estimator}"
        top_df = rdo_df.sort_values(by=metrica_col, ascending=False).head(10).copy()

        # === Gráfico Modelos ===
        ax2 = fig.add_subplot(3, 1, 3)  # Segunda fila, el gráfico
        if args.algorithm == "knn":
            top_df['Params'] = top_df.apply(
                lambda row: f"K{int(row['param_n_neighbors'])}_P{row['param_p']}_{row['param_weights']}", axis=1)
        elif args.algorithm == "decision_tree":
            top_df['Params'] = top_df.apply(lambda
                                                row: f"Criterion{row['param_criterion']}_Depth{row['param_max_depth']}_Split{row['param_min_samples_split']}_Leaf{row['param_min_samples_leaf']}",
                                            axis=1)
        elif args.algorithm == "random_forest":
            top_df['Params'] = top_df.apply(lambda
                                                row: f"N{row['param_n_estimators']}_Depth{row['param_max_depth']}_Split{row['param_min_samples_split']}_Leaf{row['param_min_samples_leaf']}_Bootstrap{row['param_bootstrap']}",
                                            axis=1)
        ax2.plot(top_df['Params'], top_df[f"mean_test_{args.estimator}"], marker='o', linestyle='-', color='b',
                 label=args.estimator.upper())

        # Configuración para que se vea bonito

        ax2.set_xticks(range(len(top_df)))
        ax2.set_xticklabels(top_df['Params'], rotation=90, fontsize=8)  # Rota los nombres en el eje X
        ax2.set_title(f"Bonanza modelos según {args.estimator.upper()}", fontsize=14, pad=20)
        ax2.set_xlabel('Combinación', fontsize=12)
        ax2.set_ylabel(f"{args.estimator.upper()}", fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.2)

        plt.savefig("Prueba", dpi=300, bbox_inches='tight')
        plt.show()