import pandas as pd

path = "exomol_data/CO/12C-16O/Li2015/12C-16O__Li2015.trans.h5"
with pd.HDFStore(path, "r") as store:
    st = store.get_storer("df")
    print("non_index_axes:", st.non_index_axes)
    print("data_columns:", getattr(st.attrs, "data_columns", None))
    print(store.select("df", start=0, stop=5))