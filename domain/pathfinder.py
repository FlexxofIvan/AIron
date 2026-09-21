# domain/pathfinder.py
"""
Поиск пути по 4-связной сетке.
Чистые функции: принимают объект с полями .width, .height, .matrix, .OBSTACLE.
Никаких зависимостей от UI, файлов и классов карты.
"""
import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal, weight=1.0):
    """
    A* с tie-breaking. weight=1.0 — оптимально; >1.0 — быстрее, но путь длиннее.
    Возвращает список координат [(y, x), ...] от start до goal или None.
    """
    for name, point in (("start", start), ("goal", goal)):
        y, x = point
        if not (0 <= y < grid.height and 0 <= x < grid.width):
            print(f"[A*] {name} {point} вне границ карты.")
            return None
        if grid.matrix[y, x] == grid.OBSTACLE:
            print(f"[A*] {name} {point} находится в препятствии.")
            return None

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = [(fscore[start], heuristic(start, goal), start)]

    while oheap:
        current_f, _, current = heapq.heappop(oheap)

        if current in close_set:
            continue
        if current_f > fscore.get(current, float("inf")):
            continue

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)

        for dy, dx in neighbors:
            neighbor = (current[0] + dy, current[1] + dx)
            if not (0 <= neighbor[0] < grid.height and 0 <= neighbor[1] < grid.width):
                continue
            if grid.matrix[neighbor] == grid.OBSTACLE:
                continue
            if neighbor in close_set:
                continue

            tentative_g = gscore[current] + int(grid.matrix[neighbor])
            if tentative_g < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + weight * heuristic(neighbor, goal)
                heapq.heappush(
                    oheap,
                    (fscore[neighbor], heuristic(neighbor, goal), neighbor),
                )
    return None


def dijkstra_from(grid, start):
    """
    Single-source Dijkstra по 4-связной сетке.
    Возвращает (came_from, dist).
    """
    dist = {start: 0}
    came_from = {}
    heap = [(0, start)]
    visited = set()

    while heap:
        d, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = current[0] + dy, current[1] + dx
            if not (0 <= ny < grid.height and 0 <= nx < grid.width):
                continue
            if grid.matrix[ny, nx] == grid.OBSTACLE:
                continue
            # int(...) обязателен: numpy.int64 не сериализуется в JSON
            nd = d + int(grid.matrix[ny, nx])
            if nd < dist.get((ny, nx), float("inf")):
                dist[(ny, nx)] = nd
                came_from[(ny, nx)] = current
                heapq.heappush(heap, (nd, (ny, nx)))

    return came_from, dist


def reconstruct_path(came_from, start, goal):
    """Восстановление пути из дерева Дейкстры."""
    if goal == start:
        return [start]
    if goal not in came_from:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    return path[::-1]