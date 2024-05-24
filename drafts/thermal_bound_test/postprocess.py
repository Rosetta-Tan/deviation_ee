import pandas as pd

LAs = [3, 4, 5, 6, 7, 8, 9, 10, 12]

# Load data
data = []
for LA in LAs:
    filename = f'data_NA={2*LA}.csv'
    try:
        if LA == 7:
            header = ['NA','seed','tr_GA_squared']
            df = pd.read_csv(filename, skiprows=[1])
            df.columns = header
        else:
            df = pd.read_csv(filename)
        data.append(df)
        avg = df['tr_GA_squared'].mean()
        print(f'NA={2*LA}, avg={avg}')
    except FileNotFoundError:
        print(f'file not found: {filename}')
    except Exception as e:
        print(f'error occurred while reading {filename}: {str(e)}')
