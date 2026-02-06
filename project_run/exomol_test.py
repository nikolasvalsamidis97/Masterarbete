from radis.io.exomol import fetch_exomol

df = fetch_exomol("H2O", 
                  database="POKAZATEL",
                  local_databases="project_run/exomol_cache",
                  load_wavenum_min=2000,
                  load_wavenum_max=5000
                  )

print(df.head)
print(len(df))
print(df.columns)