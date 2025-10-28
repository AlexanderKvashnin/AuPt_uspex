import os
import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from ase.io import read, write
from ase import Atoms

# Функция для загрузки кластера и конвертации в pymatgen Molecule
def load_cluster(filename):
    try:
        structure = Structure.from_file(filename)
        species = [site.specie for site in structure]
        coords = [site.coords for site in structure]
        molecule = Molecule(species, coords)
        return molecule
    except Exception as e:
        print(f"Ошибка при загрузке кластера: {e}")
        return None

# Функция для определения всех осей симметрии и группы симметрии
def get_axes_and_symmetry_group(molecule):
    analyzer = PointGroupAnalyzer(molecule, tolerance=0.1, eigen_tolerance=0.01)
    symmetry_group = analyzer.get_pointgroup()


    symmetry_ops = analyzer.get_symmetry_operations()

    axes_dict = {}
    for op in symmetry_ops:
        if np.isclose(np.linalg.det(op.rotation_matrix), 1.0, atol=1e-6):
            rotation_matrix = op.rotation_matrix
            cos_theta = (np.trace(rotation_matrix) - 1) / 2
            cos_theta = np.clip(cos_theta, -1, 1)
            angle = np.arccos(cos_theta)

            if angle > 0:
                calculated_order = round(2 * np.pi / angle)

                if calculated_order < 2 or calculated_order > 12:
                    continue

                eigenvalues, eigenvectors = np.linalg.eig(rotation_matrix)
                for i, eigenvalue in enumerate(eigenvalues):
                    if np.isclose(eigenvalue, 1.0, atol=1e-6):
                        axis = eigenvectors[:, i].real
                        axis /= np.linalg.norm(axis)
                        if calculated_order not in axes_dict:
                            axes_dict[calculated_order] = []
                        axes_dict[calculated_order].append(axis)

    # Удаление дублей
    unique_axes = {}
    for order, axes in axes_dict.items():
        unique_axes[order] = []
        for axis in axes:
            if not any(
                np.allclose(axis, existing_axis, atol=1e-3) or
                np.allclose(axis, -existing_axis, atol=1e-3)
                for existing_axis in unique_axes[order]
            ):
                unique_axes[order].append(axis)

    return unique_axes, symmetry_group

# Функция для записи осей симметрии и группы в текстовый файл
def write_axes_to_file(axes, symmetry_group, output_file, filename):
    with open(output_file, 'a') as f:
        f.write(f"{filename}\n")
        f.write(f"Группа симметрии: {symmetry_group}\n")
        f.write("Найденные оси симметрии:\n")
        for order, axes_list in axes.items():
            for axis in axes_list:
                f.write(f"Порядок: {order}, Ось: {axis}\n")
        f.write("\n")

# Функция для выравнивания кластера относительно подложки
def align_centers(cluster, substrate, distance):
    cluster_center = cluster.get_center_of_mass()
    substrate_center = substrate.get_center_of_mass()

    displacement = np.array([substrate_center[0] - cluster_center[0],
                             substrate_center[1] - cluster_center[1],
                             0])
    cluster.translate(displacement)

    cluster_bottom = cluster.get_positions()[:, 2].min()
    substrate_top = substrate.get_positions()[:, 2].max()
    z_shift = substrate_top - cluster_bottom + distance
    cluster.translate([0, 0, z_shift])

    return cluster

# Функция для поворота вдоль заданной оси в плоскости, определяемой осью и вектором для сравнения
def rotate_along_axis(cluster, axis, target_angle, reference_vector):
    # Нормализуем ось и вектор для сравнения
    axis = axis / np.linalg.norm(axis)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    
    # Вычисляем угол между осью и заданным вектором
    cos_current_angle = np.dot(axis, reference_vector)
    current_angle = np.arccos(np.clip(cos_current_angle, -1.0, 1.0))  # Угол в радианах
    
    # Определяем угол поворота, который нам нужен
    angle_to_rotate = target_angle - current_angle
    
    # Вычисляем векторное произведение для нахождения новой оси вращения
    rotation_axis = np.cross(axis, reference_vector)

    # Проверяем, является ли ось вращения нулевым вектором
    if np.linalg.norm(rotation_axis) == 0:
        print("Warning: rotation_axis is zero. Skipping rotation.")
        return cluster  # Можно также выбросить исключение или обработать иначе
    
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Нормализуем

    # Вычисляем косинус и синус нового угла
    cos_theta = np.cos(angle_to_rotate)
    sin_theta = np.sin(angle_to_rotate)
    ux, uy, uz = rotation_axis
    
    # Создаем матрицу поворота вокруг новой оси
    rotation_matrix = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])
    
    # Получаем позиции кластера
    positions = cluster.get_positions()
    center = cluster.get_center_of_mass()
    
    # Центрируем позиции
    positions -= center
    
    # Применяем матрицу поворота
    positions = np.dot(positions, rotation_matrix.T)
    
    # Возвращаем позиции в исходное положение
    positions += center
    cluster.set_positions(positions)
    
    return cluster

# Функция для создания конфигураций на основе условий
def create_configurations_based_on_axes(cluster, axes, substrate, output_folder, filename):
    config_count = 1  # Начинаем с номера 1 для конфигураций (нулевая будет отдельно)

    def save_configuration(cluster, name_suffix, distance):
        """Сохранение конфигурации с учетом расстояния."""
        nonlocal config_count
        aligned_cluster = align_centers(cluster, substrate, distance)
        combined = Atoms(substrate + aligned_cluster)
        output_path = os.path.join(output_folder, f"{filename}_config_{config_count}{name_suffix}.vasp")
        write(output_path, combined, format='vasp')
        config_count += 1

    def generate_random_configuration(cluster):
        """Генерация случайной конфигурации."""
        random_axis = np.random.randn(3)
        random_angle = np.random.rand() * 2 * np.pi
        return rotate_along_axis(cluster.copy(), random_axis, random_angle, np.array([1, 0, 0]))

    # Создаём нулевую конфигурацию с расстоянием 8 Å
    aligned_cluster = align_centers(cluster.copy(), substrate, distance=8)
    combined = Atoms(substrate + aligned_cluster)
    output_path = os.path.join(output_folder, f"{filename}_config_0_initial.vasp")
    write(output_path, combined, format='vasp')

    # Создаем остальные конфигурации с расстоянием 2 Å
    if 3 in axes and len(axes[3]) >= 4:  # 4 оси третьего порядка
        axis = axes[3][0]
        save_configuration(rotate_along_axis(cluster.copy(), axis, np.radians(35), np.array([0, 0, 1])), "_35deg_XYZ", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([0, 0, 1])), "_along_Z", distance=2.5)

    elif 6 in axes:  # Ось шестого порядка
        axis = axes[6][0]
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([0, 0, 1])), "_along_Z", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([1, 0, 0])), "_along_X", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, np.radians(30), np.array([0, 0, 1])), "_30deg", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, np.radians(120), np.array([0, 0, 1])), "_120deg", distance=2.5)

    elif 3 in axes and len(axes[3]) == 1:  # 1 ось третьего порядка
        axis = axes[3][0]
        save_configuration(rotate_along_axis(cluster.copy(), axis, 1.6, np.array([1, 1, 0])), "_along_Z", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0.1, np.array([1, 0, 0])), "_along_X", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0.5, np.array([0, 1, 1])), "_30deg", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 2.09, np.array([0, 1, 1])), "_120deg", distance=2.5)

    elif 4 in axes and len(axes[4]) == 1:  # 1 ось четвертого порядка
        axis = axes[4][0]
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([0, 0, 1])), "_along_Z", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([1, 0, 0])), "_along_X", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, np.radians(30), np.array([0, 0, 1])), "_30deg", distance=2.5)

    elif 2 in axes and len(axes[2]) == 3:  # 3 оси второго порядка
        axis = axes[2][0]
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([0, 0, 1])), "_along_Z", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([1, 0, 0])), "_along_X", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0.5, np.array([1, 0, 0])), "_30deg", distance=2.5)
        for _ in range(1):  # Генерация 1 случайных конфигураций
            save_configuration(generate_random_configuration(cluster), "_random", distance=2)

    elif 2 in axes and len(axes[2]) == 1:  # 1 ось второго порядка
        axis = axes[2][0]
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([0, 0, 1])), "_along_Z", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([1, 0, 0])), "_along_X", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0.8, np.array([1, 1, 0])), "_45deg", distance=2.5)
        for _ in range(2):  # Генерация 2 случайных конфигураций
            save_configuration(generate_random_configuration(cluster), "_random", distance=2.5)

    elif any(order >= 7 or order == 5 for order in axes):  # Оси порядка 5, 7 и выше
        order = max(axes.keys())
        axis = axes[order][0]
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([0, 0, 1])), "_along_Z", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0, np.array([1, 0, 0])), "_along_X", distance=2.5)
        save_configuration(rotate_along_axis(cluster.copy(), axis, 0.5, np.array([1, 0, 0])), "_30deg", distance=2.5)



    if not axes:  # Нет осей
        for _ in range(5):  # Генерация 5 случайных конфигураций
            save_configuration(generate_random_configuration(cluster), "_random", distance=2.5)
        

# Основная программа
clusters_folder = "D:\\work\\Radina\\Clusters"
output_folder = "D:\\work\\Radina\\done"
substrate_file = "D:\\work\\Radina\\CONTCAR-Gr"
axes_output_file = "D:\\work\\Radina\\axes_info.txt"  # Файл для записи информации о осях

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Считываем подложку
substrate = read(substrate_file)

# Очищаем файл перед записью
if os.path.exists(axes_output_file):
    os.remove(axes_output_file)

# Проходимся по файлам кластеров
for cluster_file in os.listdir(clusters_folder):
    if cluster_file.endswith(".vasp"):
        cluster_path = os.path.join(clusters_folder, cluster_file)
        filename = os.path.splitext(cluster_file)[0]

        # Загрузка кластера и определение осей симметрии и группы симметрии
        cluster = read(cluster_path)
        molecule = load_cluster(cluster_path)
        if molecule is None:
            continue

        axes, symmetry_group = get_axes_and_symmetry_group(molecule)
        print(f"Обработка {cluster_file}: группа симметрии {symmetry_group}, найдено осей симметрии {axes}")

        # Запись информации о осях и группе симметрии в файл
        write_axes_to_file(axes, symmetry_group, axes_output_file, filename)

        # Создаем конфигурации
        create_configurations_based_on_axes(cluster, axes, substrate, output_folder, filename)
