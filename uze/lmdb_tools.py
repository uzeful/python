import lmdb

def get_lmdb_value(db_env, key, iswrite=False):
    # db_env is the lmdb environment, iswrite is boolean object, indicating whether to open db for writting
    with db_env.begin(write=iswrite) as txn:
        with txn.cursor() as cursor:
            return cursor.get(key)

def lmdb_put(db_env, key, datum, iswrite=True):
    with db_env.begin(write=iswrite) as txn:
        txn.put(key, datum)

def get_lmdb_key(lmdb_path, keyfile_path):
    # open lmdb data base
    env = lmdb.open(lmdb_path)
    # open keyfile for write
    keyfile = open(keyfile_path, 'w')
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for key, value in cursor:
                keyfile.write(key)
                keyfile.write('\n')
    env.close()
