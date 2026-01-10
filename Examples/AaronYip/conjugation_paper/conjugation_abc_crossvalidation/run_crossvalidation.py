import os

# Runs ABC calibration on all 5 folds. Deletes simulation data after each run to ensure we don't run out of storage

for i in [2,3,4,5]:
    os.chdir(f'/home/user/CellModeller/Examples/AaronYip/conjugation_paper/conjugation_abc_crossvalidation/{i}')
    os.system(f"python abc_conjugation_redis_equal{i}.py")
    print()
    os.system("rm -rf data")
