import helpers.pre_process as pre_process


df = pre_process.get_processed_data()


# Print 1 row of data
# for c in df.columns:
#     print('{}:\t{}'.format(c, df[c][0]))

# Print unique values: ingredient_unit
# print(df.unit.unique().sort())
# print(sorted(df.unit.astype(str).unique()))
# print(df[df.unit == 'large cloves'])



# df['comment'] = df.comment.map(lambda x: x.upper() if x else None)
# for i in range(5):
#     print(df.iloc[i])

print(df.describe())