import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')


def pro_data_analysis():
    file_path = 'Data_Set_5.xlsx'
    try:
        df = pd.read_excel(file_path, sheet_name='Sales Data')
    except Exception as e:
        print(f"Помилка: {e}")
        return

    df = df.dropna(subset=['OrderDate', 'Region', 'Sale_amt', 'Units', 'Unit_price'])
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df = df.sort_values('OrderDate')

    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Розширене R&D дослідження показників ефективності (Data_Set_5)', fontsize=18, fontweight='bold',
                 y=0.95)

    # Графік 1: Ковзне середнє
    ax1 = fig.add_subplot(2, 2, 1)
    df_time = df.groupby('OrderDate')['Sale_amt'].sum().reset_index()
    df_time['SMA_3'] = df_time['Sale_amt'].rolling(window=3, min_periods=1).mean()

    ax1.plot(df_time['OrderDate'], df_time['Sale_amt'], marker='o', alpha=0.4, label='Фактичні щоденні продажі',
             color='blue')
    ax1.plot(df_time['OrderDate'], df_time['SMA_3'], linewidth=3, label='Тренд (Ковзне середнє, вікно=3)', color='red')
    ax1.set_title('Графік 1: Аналіз справжнього тренду (Ковзне середнє)')
    ax1.set_ylabel('Обсяг продажів')
    ax1.legend()

    # Графік 2: Аналіз розподілу та викидів (Boxplot)
    ax2 = fig.add_subplot(2, 2, 2)
    sns.boxplot(x='Region', y='Sale_amt', data=df, ax=ax2, hue='Region', palette='Set2', legend=False)
    sns.stripplot(x='Region', y='Sale_amt', data=df, ax=ax2, color=".3", alpha=0.5, size=6)  # Додаємо точки поверх
    ax2.set_title('Графік 2: Статистичний розподіл продажів та пошук аномалій')
    ax2.set_ylabel('Сума окремої транзакції')

    # Графік 3: Кореляційна матриця Пірсона (Heatmap)
    ax3 = fig.add_subplot(2, 2, 3)
    # Відбираємо лише числові колонки
    num_cols = df[['Units', 'Unit_price', 'Sale_amt']]
    corr_matrix = num_cols.corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax3, vmin=-1, vmax=1)
    ax3.set_title('Графік 3: Кореляційна матриця (Вплив факторів на дохід)')

    # Графік 4: Крос-аналіз менеджерів за регіонами
    ax4 = fig.add_subplot(2, 2, 4)
    cross_tab = pd.crosstab(index=df['Manager'], columns=df['Region'], values=df['Sale_amt'], aggfunc='sum').fillna(0)

    cross_tab.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis')
    ax4.set_title('Графік 4: Структура доходів: Менеджери vs Регіони')
    ax4.set_xlabel('Менеджер')
    ax4.set_ylabel('Сумарний обсяг продажів')
    ax4.legend(title='Регіон')
    ax4.tick_params(axis='x', rotation=0)

    # Фіналізація
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('Analys.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    pro_data_analysis()