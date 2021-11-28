import nglview
import tempfile



def fix_PDB_element_column(PDB_file):
    """
    Some modelling packages do not write the element column in the PDB.
    This function returns a fixed PDB string with correct values for the element column.

    Assigns each atom's element based on the atom name field.
    Rules specified in the format specification:
        - Alignment of one-letter atom name such as C starts at column 14,
          while two-letter atom name such as FE starts at column 13.
        - Atom nomenclature begins with atom type.
    adapted from : https://github.com/haddocking/pdb-tools/blob/master/pdbtools/pdb_element.py
    """
    def pad_line(line):
        """
        Helper function to pad line to 80 characters in case it is shorter
        """
        size_of_line = len(line)
        if size_of_line < 80:
            padding = 80 - size_of_line + 1
            line = line.strip('\n') + ' ' * padding + '\n'
        return line[:81]  # 80 + newline character

    elements = set(
        ('H', 'D', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE', 'NA', 'MG',
         'AL', 'SI', 'P', 'S', 'CL', 'AR', 'K', 'CA', 'SC', 'TI', 'V', 'CR',
         'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'GE', 'AS', 'SE', 'BR',
         'KR', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG',
         'CD', 'IN', 'SN', 'SB', 'TE', 'I', 'XE', 'CS', 'BA', 'LA', 'CE', 'PR',
         'ND', 'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB',
         'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB',
         'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH', 'PA', 'U', 'NP', 'PU',
         'AM', 'CM', 'BK', 'CF', 'ES', 'FM', 'MD', 'NO', 'LR', 'RF', 'DB',
         'SG', 'BH', 'HS', 'MT'))
    records = ('ATOM', 'HETATM', 'ANISOU')
    ret = ''
    with open(PDB_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = pad_line(line.strip('\n'))
        if line.startswith(records):
            line = pad_line(line)
            atom_name = line[12:16]
            if atom_name[0].isalpha() and not atom_name[2:].isdigit():
                element = atom_name.strip()
            else:
                atom_name = atom_name.strip()
                if atom_name[0].isdigit():
                    element = atom_name[1]
                else:
                    element = atom_name[0]
            if element not in elements:
                element = '  '  # empty element in case we cannot assign
                print(
                    'fix_PDB_element_column(): WARNING, cannot assign element.'
                )
            line = line[:76] + element.rjust(2) + line[78:]
        ret += line + '\n'
    return ret


def visualise(mol, coordinate_system=None):
    '''A helper function to visualise an ampal object in the notebook using nglview
    '''
    view = None
    with tempfile.NamedTemporaryFile(delete=True, suffix='.pdb') as tmp:
        tmp.write(mol.pdb.encode())
        tmp.seek(0)  # Resets the buffer back to the first line
        view = nglview.show_file(tmp.name)
    if coordinate_system is not None:
        view = add_coordinate_system_to_view(view, coordinate_system)
    return view

def add_coordinate_system_to_view(view, coordinate_system):
    for v, name, color in zip(coordinate_system,
                              ['x', 'y', 'z'],
                              [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        view.shape.add_arrow([0, 0, 0], 5*v, color, 0.5, name)
    return view