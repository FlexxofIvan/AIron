# map.py
import math
import numpy as np
import matplotlib.pyplot as plt
import main_rover

from matplotlib.colors import ListedColormap

from domain.commands import angle_from_heading
from services.robot_driver import RobotDriver
from services import route_service
from services.mission import Mission, MissionStatus
from domain.constants import AUTO_RETURN_WAIT_SEC

class InteractiveMapUI:
    def __init__(self, grid, commands_dir, robot_pos=(1, 1),
                 robot_home=(1, 1), robot_heading="NORTH",
                 routes_database=None):
        self.grid            = grid
        self.commands_dir    = commands_dir
        self.routes_database = routes_database or {}
        self.driver          = RobotDriver(main_rover.run, main_rover.open_session)

        self.robot_pos     = tuple(robot_pos)
        self.robot_home    = tuple(robot_home)
        self.robot_heading = robot_heading
        self.current_visual_path = None

        # --- трекер и анимация ---
        self.mission      = None
        self._timer       = None       
        self._wait_timer  = None       
        self._draw_x      = float(self.robot_pos[1])
        self._draw_y      = float(self.robot_pos[0])
        self._draw_angle  = angle_from_heading(self.robot_heading)

    # ---------- Интерактивный интерфейс ----------

    def show_interactive_map(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        try:
            self.fig.canvas.manager.set_window_title('Панель управления Rover Revolution')
        except Exception:
            pass

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._setup_timer()
        self._update_plot()
        plt.show()

    def _setup_timer(self):
        """Создаёт таймер на 20 FPS. Если бэкенд не поддерживает — тихо пропускаем."""
        try:
            self._timer = self.fig.canvas.new_timer(interval=50)
            self._timer.add_callback(self._on_tick)
        except (AttributeError, NotImplementedError) as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Таймер недоступен: {e}")
            self._timer = None

    def _on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Идёт миссия — игнорируем
        if self.mission is not None:
            print("[ИНТЕРФЕЙС] Робот занят, дождитесь завершения.")
            return

        # Любой клик отменяет запланированный авто-возврат
        self._cancel_auto_return()

        if event.button == 3:        # правый клик — домой
            self._start_return_home()
            return

        click_x = int(round(event.xdata))
        click_y = int(round(event.ydata))

        clicked_target_id = self.grid.obs_at(click_y, click_x)
        if clicked_target_id is None:
            print(f"[ИНТЕРФЕЙС] Клик мимо тренажера в точку ({click_y}, {click_x}).")
            return

        print(f"\n[ИНТЕРФЕЙС] Клик по Тренажеру #{clicked_target_id}!")

        target_pos = tuple(self.grid.slots[clicked_target_id])
        result = route_service.build_route_to_target(
            self.grid, self.robot_pos, self.robot_heading,
            target_pos, self.commands_dir,
        )
        if result is None:
            print(f"[ОШИБКА] Путь к тренажёру #{clicked_target_id} не найден.")
            return

        path, commands, final_heading, cmd_file = result

        self.current_visual_path = path
        self._launch_mission(cmd_file, commands, target_pos, final_heading)
        
    def _start_return_home(self):
        result = route_service.build_return_home_route(
            self.grid, self.robot_home,
            self.robot_pos, self.robot_heading,
            self.commands_dir,
        )
        if result is None:
            return
        path, commands, final_heading, cmd_file = result

        self.current_visual_path = path
        self._launch_mission(cmd_file, commands, self.robot_home, final_heading,
                             is_return_home=True)
        
    def _launch_mission(self, command_file, commands, target_pos, target_heading,
                        is_return_home=False):
        """Создаёт Mission, запускает её и включает таймер обновления."""
        self.mission = Mission(
            commands=commands,
            start_pos=self.robot_pos,
            start_heading=self.robot_heading,
            driver=self.driver,
            target_pos=target_pos,
            target_heading=target_heading,
            command_file=command_file,
            is_return_home=is_return_home,
        )
        print(f"[РОБОТ] Старт миссии. Ожидаемое время: {self.mission.total_time:.2f} с.")
        self.mission.start()

        if self._timer is not None:
            self._timer.start()

    def _on_tick(self):
        """Вызывается 20 раз в секунду, пока активна миссия."""
        if not self.mission:
            return

        s = self.mission.state_now()
        if s is not None:
            self._draw_x, self._draw_y, self._draw_angle = s
            self._update_plot()

        if self.mission.is_finished():
            self._finalize_mission()

    def _finalize_mission(self):
        """Миссия завершена: фиксируем позицию, гасим таймер, планируем возврат."""
        if self._timer is not None:
            self._timer.stop()

        was_return_home = self.mission.is_return_home
        status = self.mission.finalize()

        if status == MissionStatus.DONE:
            self.robot_pos     = self.mission.target_pos
            self.robot_heading = self.mission.target_heading
            print(f"[РОБОТ] Прибыл в {self.robot_pos}, курс {self.robot_heading}.")
        else:
            print(f"[РОБОТ] Миссия прервана. Робот остался в {self.robot_pos}.")

        self.current_visual_path = None
        self._draw_x     = float(self.robot_pos[1])
        self._draw_y     = float(self.robot_pos[0])
        self._draw_angle = angle_from_heading(self.robot_heading)

        self.mission = None
        self._update_plot()

        # Успешно приехали к тренажёру и не на доке → ждём новых указаний
        if (status == MissionStatus.DONE
                and not was_return_home
                and tuple(self.robot_pos) != tuple(self.robot_home)):
            self._schedule_auto_return()

    def _schedule_auto_return(self):
        """Заводит one-shot таймер: если за N секунд не будет кликов — едем домой."""
        if self._wait_timer is None:
            try:
                self._wait_timer = self.fig.canvas.new_timer(
                    interval=int(AUTO_RETURN_WAIT_SEC * 1000)
                )
                self._wait_timer.add_callback(self._on_auto_return)
            except (AttributeError, NotImplementedError) as e:
                print(f"[ПРЕДУПРЕЖДЕНИЕ] Авто-возврат недоступен: {e}")
                return

        print(f"[РОБОТ] Жду новых указаний {AUTO_RETURN_WAIT_SEC:.0f} с...")
        self._wait_timer.start()

    def _cancel_auto_return(self):
        """Отменяет запланированный авто-возврат (если он был)."""
        if self._wait_timer is not None:
            self._wait_timer.stop()

    def _on_auto_return(self):
        """Сработал таймер ожидания: если ничего не запущено — едем домой."""
        if self._wait_timer is not None:
            self._wait_timer.stop()

        if self.mission is not None:
            return

        if tuple(self.robot_pos) == tuple(self.robot_home):
            return

        print("[РОБОТ] Новых указаний нет — возвращаюсь на док.")
        self._start_return_home()

    def _update_plot(self):

        vis_matrix = np.zeros_like(self.grid.matrix)
        vis_matrix[self.grid.matrix == self.grid.PRIORITY]     = 0
        vis_matrix[self.grid.matrix == self.grid.DEFAULT_FREE] = 1
        vis_matrix[self.grid.matrix == self.grid.OBSTACLE]     = 2

        if self.current_visual_path:
            for y, x in self.current_visual_path:
                vis_matrix[y, x] = 3


        for s_y, s_x in self.grid.slots.values():
            if vis_matrix[s_y, s_x] != 3:
                vis_matrix[s_y, s_x] = 5

        custom_cmap = ListedColormap(
            ['#a1dbcd', '#f0f0f0', '#2c3e50', '#ff4d4d', '#4a90e2', '#9b59b6']
        )

        self.ax.clear()
        self.ax.imshow(vis_matrix, cmap=custom_cmap, origin='upper')
        self.ax.grid(True, color='white', linewidth=0.5)
        self.ax.set_title("ЛКМ по тренажёру — ехать к нему | ПКМ — вернуться на док",
                          fontsize=11)

        self._draw_robot_arrow()

        # Реальное обновление окна
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _draw_robot_arrow(self):
        """Рисует стрелку-робот в непрерывных координатах с учётом курса."""
        x, y = self._draw_x, self._draw_y
        rad  = math.radians(self._draw_angle)
        dx   =  math.sin(rad)    # восток: +
        dy   = -math.cos(rad)    # север:  −y (origin='upper')

        L    = 0.55
        tail = (x - dx * L, y - dy * L)
        head = (x + dx * L, y + dy * L)

        self.ax.annotate(
            "",
            xy=head, xytext=tail,
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c",
                            lw=3, mutation_scale=22),
            zorder=10,
        )
        self.ax.plot(x, y, marker='o', color='#e74c3c',
                     markersize=4, zorder=11)
