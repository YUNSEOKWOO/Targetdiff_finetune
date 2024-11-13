import lmdb
import pickle
import json

def read_lmdb(lmdb_path, num_entries=None, output_file='lmdb_output.json'):
    """
    Reads the LMDB file and saves the contents to a JSON file.
    
    Args:
        lmdb_path (str): Path to the LMDB file.
        num_entries (int, optional): Number of entries to read. If None, reads all entries.
        output_file (str): Path to save the output JSON file.
    
    Returns:
        data_list (list): List of deserialized objects from the LMDB file.
    """
    data_list = []
    
    # Open LMDB file (single file, not a directory)
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    
    with env.begin() as txn:
        # **Added Lines Start**
        # Retrieve LMDB statistics
        stat = txn.stat()
        total_keys = stat['entries']
        print(f"Total number of keys in LMDB: {total_keys}")
        # **Added Lines End**
        
        cursor = txn.cursor()
        
        for i, (key, value) in enumerate(cursor):
            if num_entries and i >= num_entries:
                break
            # Deserialize the value
            data = pickle.loads(value)
            
            # Convert the key-value pair to a dictionary for saving
            data_entry = {
                'key': key.decode(),   # The key as a string
                'data': data           # The deserialized data
            }
            data_list.append(data_entry)
    
    # Save the list of key-value pairs to a JSON file
    with open(output_file, 'w') as f:
        # Convert the list of objects to JSON (ensure objects are serializable)
        json.dump(data_list, f, indent=4, default=str)  # `default=str` handles non-serializable objects
    
    print(f"Saved output to {output_file}")
    return data_list

if __name__ == "__main__":
    # lmdb_path = 'data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
    # lmdb_path = 'data/output_ligand_protein.lmdb'
    lmdb_path = 'data/mpro_docked_ligands_7uup_pocket_final.lmdb/data.mdb'
    
    # Read and save the first 10 entries from the LMDB file
    data_list = read_lmdb(lmdb_path, num_entries=10, output_file='lmdb_output_ours_7uup_pocket.json')

    # **Optional: Count Files in Directory**
    # Uncomment the following lines if you also want to count files in a directory
    # import os, os.path
    # DIR = 'data/docked_ligands'
    # print(f"Number of files in {DIR}: {len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])}")


# import os, os.path
# DIR = 'data/docked_ligands'
# print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))