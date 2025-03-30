# -*- coding: utf-8 -*-
"""
终极生存模拟器 v7.1
完整优化版：
1. 优化性能，默认2000生物
2. 动态屏幕适配
3. 滚轮缩放功能
4. 调整生物成长效果
5. 完整战斗和繁殖系统
"""

import sys
import os
import warnings
import numpy as np
import torch
from PyQt5.QtCore import QTimer, Qt, QPointF, QRectF
from PyQt5.QtGui import (QPainter, QColor, QPen, QBrush, 
                        QRadialGradient, QFont, QTransform)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSlider,
                             QSpinBox, QGroupBox, QFrame, QDesktopWidget)

# 环境配置
warnings.filterwarnings("ignore")
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.*.debug=false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class SafeSimulator:
    MAX_DRAW_ENTITIES = 8000
    MAX_FRUITS = 3000
    INITIAL_ENERGY = 100.0
    BASE_ENERGY_DECAY = 5.0
    CATEGORY_TRAITS = {
        0: {'color': (0, 160, 255), 'speed': 1.2, 'decay': 4.0, 'gain': 1.2, 'attack': 0.0},
        1: {'color': (255, 80, 0), 'speed': 1.5, 'decay': 5.0, 'gain': 1.0, 'attack': 0.4},
        2: {'color': (50, 255, 100), 'speed': 1.2, 'decay': 3.5, 'gain': 1.8, 'attack': 0.0}
    }
    AUTO_FRUIT_INTERVAL = 0.5
    FRUIT_SPAWN_RATE = 20
    COMBAT_RANGE = 0.04
    FEEDING_RANGE = 0.035

    def __init__(self, device_type, population=2000):
        self.device = self._init_device(device_type)
        self.population = min(population, 20000 if self.device.type == 'cuda' else 5000)
        self.paused = False
        self.boundary_deaths = 0
        self.fruit_spawn_timer = 0.0
        self.reset_ecosystem()

    def _init_device(self, device_type):
        try:
            device = torch.device(device_type)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA不可用")
            torch.zeros(1, device=device)
            return device
        except Exception as e:
            warnings.warn(f"设备初始化失败: {str(e)}, 回退到CPU")
            return torch.device('cpu')

    def reset_ecosystem(self):
        try:
            self.positions = torch.clamp(
                torch.rand((2, self.population), device=self.device) * 0.8 + 0.1,
                0.1, 0.9
            )
            self.velocities = torch.randn((2, self.population), device=self.device) * 0.05
            self.energy = torch.full((self.population,), self.INITIAL_ENERGY, device=self.device)
            self.scores = torch.zeros(self.population, device=self.device)
            self.generations = torch.zeros(self.population, device=self.device)
            self.age = torch.zeros(self.population, device=self.device)
            self.max_age = torch.randint(50, 200, (self.population,), device=self.device)
            self.category = torch.randint(0, 3, (self.population,), device=self.device)
            self.alive = torch.ones(self.population, dtype=torch.bool, device=self.device)
            
            self.fruit_positions = torch.zeros((2, 0), device=self.device)
            self.fruit_values = torch.zeros(0, device=self.device)
            self.boundary_deaths = 0
        except Exception as e:
            raise RuntimeError(f"重置失败: {str(e)}")

    def spawn_fruit(self, positions, values):
        try:
            if positions.numel() == 0:
                return

            if positions.dim() == 1:
                positions = positions.unsqueeze(1)
            
            valid_mask = (positions[0] >= 0) & (positions[0] <= 1.0) & \
                         (positions[1] >= 0) & (positions[1] <= 1.0)
            valid_pos = positions[:, valid_mask]
            valid_vals = values.expand_as(valid_pos[0]) if values.numel() == 1 else values[valid_mask]

            if valid_pos.shape[1] > 0:
                combined_pos = torch.cat([self.fruit_positions, valid_pos], dim=1)
                combined_vals = torch.cat([self.fruit_values, valid_vals])
                
                if combined_pos.shape[1] > self.MAX_FRUITS:
                    keep_idx = torch.randperm(combined_pos.shape[1])[:self.MAX_FRUITS]
                    self.fruit_positions = combined_pos[:, keep_idx]
                    self.fruit_values = combined_vals[keep_idx]
                else:
                    self.fruit_positions = combined_pos
                    self.fruit_values = combined_vals
        except Exception as e:
            warnings.warn(f"果实生成失败: {str(e)}")

    def update_ecosystem(self, dt):
        if self.paused:
            return

        try:
            # 自动生成果实
            self.fruit_spawn_timer += dt
            if self.fruit_spawn_timer >= self.AUTO_FRUIT_INTERVAL:
                self._auto_spawn_fruits()
                self.fruit_spawn_timer = 0.0

            active = self.alive.clone()
            self.age[active] += 1

            # 死亡检测
            death_mask = (
                (self.age > self.max_age) |
                (self.positions[0] < 0.01) | (self.positions[0] > 0.99) |
                (self.positions[1] < 0.01) | (self.positions[1] > 0.99) |
                (self.energy <= 0)
            )
            if death_mask.any():
                self._process_death(death_mask)

            # 能量消耗
            if active.any():
                hunger = (self.INITIAL_ENERGY - self.energy[active]) / self.INITIAL_ENERGY
                traits = torch.tensor(
                    [self.CATEGORY_TRAITS[c.item()]['decay'] for c in self.category[active]],
                    device=self.device
                )
                energy_loss = self.BASE_ENERGY_DECAY * (1.0 + 2.0 * hunger) * dt
                self.energy[active] -= torch.rand_like(self.energy[active]) * (traits * energy_loss)
                self.energy[active] = torch.clamp(self.energy[active], 0.0, 200.0)

            # 生物行为
            if active.any():
                self._update_movement(dt, active)
                self._update_feeding(active)
                self._update_combat(active)
                self._update_reproduction(active)
        except Exception as e:
            warnings.warn(f"更新失败: {str(e)}")

    def _auto_spawn_fruits(self):
        new_pos = torch.rand((2, self.FRUIT_SPAWN_RATE), device=self.device)
        new_vals = torch.full((self.FRUIT_SPAWN_RATE,), 50.0, device=self.device)
        self.spawn_fruit(new_pos, new_vals)

    def _process_death(self, dead_mask):
        try:
            dead_pos = self.positions[:, dead_mask]
            self.spawn_fruit(dead_pos, self.scores[dead_mask] * 0.8)
            
            self.alive[dead_mask] = False
            self.scores[dead_mask] = 0
            self.generations[dead_mask] = 0
            self.age[dead_mask] = 0
            
            boundary_death = (
                (self.positions[0][dead_mask] < 0.01) | 
                (self.positions[0][dead_mask] > 0.99) |
                (self.positions[1][dead_mask] < 0.01) | 
                (self.positions[1][dead_mask] > 0.99)
            )
            self.boundary_deaths += boundary_death.sum().item()
        except Exception as e:
            warnings.warn(f"死亡处理失败: {str(e)}")

    def _update_movement(self, dt, active):
        try:
            hunger = (self.INITIAL_ENERGY - self.energy[active]) / self.INITIAL_ENERGY
            base_speeds = torch.tensor(
                [self.CATEGORY_TRAITS[c.item()]['speed'] for c in self.category[active]],
                device=self.device
            )
            speeds = (base_speeds * (1.0 + 0.8 * hunger)).unsqueeze(0)

            if self.fruit_positions.shape[1] > 0:
                pos = self.positions[:, active].unsqueeze(2)
                fruits = self.fruit_positions.unsqueeze(1)
                distances = torch.norm(fruits - pos, dim=0)
                closest = torch.argmin(distances, dim=1)
                
                move_dir = (fruits - pos)[:, torch.arange(pos.shape[1]), closest]
                move_strength = 0.035 * (1.0 + 3.0 * hunger)
                self.velocities[:, active] += move_dir * move_strength * dt * speeds

            rand_strength = 0.008 * (1.0 - 0.9 * hunger)
            self.velocities[:, active] = torch.clamp(
                self.velocities[:, active] + torch.randn_like(self.velocities[:, active]) * rand_strength,
                -0.7, 0.7
            )
            self.positions[:, active] = torch.clamp(
                self.positions[:, active] + self.velocities[:, active] * dt,
                0.01, 0.99
            )
        except Exception as e:
            warnings.warn(f"移动更新失败: {str(e)}")

    def _update_feeding(self, active):
        try:
            if self.fruit_positions.shape[1] == 0:
                return

            active_indices = torch.where(active)[0]
            pos = self.positions[:, active_indices].unsqueeze(2)
            fruits = self.fruit_positions.unsqueeze(1)
            
            distances = torch.norm(pos - fruits, dim=0)
            closest_dist = distances.min(dim=1).values
            hunger = (self.INITIAL_ENERGY - self.energy[active_indices]) / self.INITIAL_ENERGY
            detect_range = self.FEEDING_RANGE * (1.0 + 0.8 * hunger)
            in_range = closest_dist < detect_range

            if in_range.any():
                eaters = active_indices[in_range]
                fruits_eaten = distances[in_range].argmin(dim=1)

                gains = torch.tensor(
                    [self.CATEGORY_TRAITS[c.item()]['gain'] for c in self.category[eaters]],
                    device=self.device
                )
                energy_gain = self.fruit_values[fruits_eaten] * gains
                
                # 调整成长效果：积分增长更明显，能量增长适度
                self.energy[eaters] += energy_gain * 10  # 原15
                self.scores[eaters] += energy_gain * 12  # 原8，使体型变化更明显

                # 移除被吃果实
                unique_fruits = torch.unique(fruits_eaten)
                mask = torch.ones(self.fruit_positions.shape[1], dtype=torch.bool, device=self.device)
                mask[unique_fruits] = False
                self.fruit_positions = self.fruit_positions[:, mask]
                self.fruit_values = self.fruit_values[mask]
        except Exception as e:
            warnings.warn(f"进食更新失败: {str(e)}")

    def _update_combat(self, active):
        try:
            active_indices = torch.where(active)[0]
            if len(active_indices) < 2:
                return

            # 掠食型生物索引
            predators = active_indices[self.category[active_indices] == 1]
            if len(predators) == 0:
                return

            # 计算距离矩阵
            pred_pos = self.positions[:, predators].T.unsqueeze(1)
            other_pos = self.positions[:, active_indices].T.unsqueeze(0)
            dist = torch.norm(pred_pos - other_pos, dim=2)

            # 寻找攻击目标（非同类且在攻击范围内）
            valid_targets = (dist < self.COMBAT_RANGE) & \
                           (self.category[active_indices] != 1).unsqueeze(0) & \
                           (active_indices != predators.unsqueeze(1))

            for i, pred_idx in enumerate(predators):
                targets = active_indices[valid_targets[i]]
                if len(targets) == 0:
                    continue

                # 选择能量最低的目标
                target_energies = self.energy[targets]
                target_idx = targets[torch.argmin(target_energies)]

                # 计算伤害
                damage = self.CATEGORY_TRAITS[1]['attack'] * self.energy[pred_idx]
                self.energy[pred_idx] += damage
                self.energy[target_idx] = torch.clamp(self.energy[target_idx] - damage, 0.0, 200.0)

                if self.energy[target_idx] <= 0:
                    self.alive[target_idx] = False
                    self._process_death(torch.tensor([target_idx]))
        except Exception as e:
            warnings.warn(f"战斗更新失败: {str(e)}")

    def _update_reproduction(self, active):
        try:
            repro_mask = (self.energy > 80) & (self.age > 30) & active
            if not repro_mask.any():
                return

            parents = torch.where(repro_mask)[0]
            child_count = parents.size(0)
            
            parent_cat = self.category[parents]
            child_cat = torch.where(
                torch.rand(child_count, device=self.device) < 0.05,
                torch.randint(0, 3, (child_count,), device=self.device),
                parent_cat
            )

            child_pos = torch.clamp(
                self.positions[:, parents] + torch.randn((2, child_count), device=self.device) * 0.03,
                0.05, 0.95
            )
            child_vel = self.velocities[:, parents] * (0.6 + torch.rand((1, child_count), device=self.device) * 0.8)
            
            self.positions = torch.cat([self.positions, child_pos], dim=1)
            self.velocities = torch.cat([self.velocities, child_vel], dim=1)
            self.energy = torch.cat([self.energy, torch.full((child_count,), 80.0, device=self.device)])
            self.scores = torch.cat([self.scores, torch.zeros(child_count, device=self.device)])
            self.generations = torch.cat([self.generations, self.generations[parents] + 1])
            self.age = torch.cat([self.age, torch.zeros(child_count, device=self.device)])
            self.max_age = torch.cat([self.max_age, torch.randint(60, 220, (child_count,), device=self.device)])
            self.category = torch.cat([self.category, child_cat])
            self.alive = torch.cat([self.alive, torch.ones(child_count, dtype=torch.bool, device=self.device)])

            self.energy[parents] -= 50
            self.population += child_count
        except Exception as e:
            warnings.warn(f"繁殖更新失败: {str(e)}")

class SafeVisualizer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        
        # 动态适配屏幕尺寸
        screen = QDesktopWidget().screenGeometry()
        self.world_size = int(min(screen.height()*0.8, 1200))
        self.setFixedSize(self.world_size, self.world_size)
        
        # 视图参数
        self.zoom_factor = 1.0
        self.view_offset = QPointF(0, 0)
        self.last_mouse_pos = None
        self.click_animation = 0.0
        self.dragging = False

    def wheelEvent(self, event):
        # 缩放控制
        zoom_in = event.angleDelta().y() > 0
        old_zoom = self.zoom_factor
        
        # 计算鼠标相对位置
        mouse_pos = event.pos()
        old_pos = self._screen_to_world(mouse_pos)
        
        # 应用缩放
        self.zoom_factor *= 1.2 if zoom_in else 0.8
        self.zoom_factor = np.clip(self.zoom_factor, 0.5, 5.0)
        
        # 调整偏移保持鼠标点位置不变
        new_pos = self._screen_to_world(mouse_pos)
        self.view_offset += (new_pos - old_pos) * old_zoom
        
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.parent.simulator.paused:
                # 转换点击位置到世界坐标
                world_pos = self._screen_to_world(event.pos())
                norm_pos = torch.tensor([
                    world_pos.x() / self.width(),
                    world_pos.y() / self.height()
                ], device=self.parent.simulator.device)
                
                self.parent.simulator.spawn_fruit(
                    norm_pos.unsqueeze(1),
                    torch.tensor([np.random.uniform(100, 150)], device=self.parent.simulator.device)
                )
                
                self.last_click_pos = event.pos()
                self.click_animation = 1.0
                QTimer.singleShot(100, self._update_animation)
        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.view_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.dragging = False
            self.last_mouse_pos = None

    def _screen_to_world(self, point):
        return QPointF(
            (point.x() - self.view_offset.x()) / self.zoom_factor,
            (point.y() - self.view_offset.y()) / self.zoom_factor
        )

    def _world_to_screen(self, point):
        return QPointF(
            point.x() * self.zoom_factor + self.view_offset.x(),
            point.y() * self.zoom_factor + self.view_offset.y()
        )

    def _update_animation(self):
        self.click_animation -= 0.1
        if self.click_animation > 0:
            self.update()
            QTimer.singleShot(50, self._update_animation)
        else:
            self.last_click_pos = None

    def paintEvent(self, event):
        qp = QPainter(self)
        try:
            # 应用变换矩阵
            transform = QTransform()
            transform.translate(self.view_offset.x(), self.view_offset.y())
            transform.scale(self.zoom_factor, self.zoom_factor)
            qp.setTransform(transform)
            
            # 只绘制可见区域
            visible_rect = transform.inverted()[0].mapRect(QRectF(self.rect()))
            
            self._draw_ecosystem(qp, visible_rect)
            self._draw_click_effect(qp)
            
            # 绘制缩放比例提示
            qp.resetTransform()
            qp.setPen(QColor(255, 255, 255))
            qp.setFont(QFont("Arial", 10))
            qp.drawText(10, 20, f"缩放: {self.zoom_factor:.1f}x")
        except Exception as e:
            warnings.warn(f"渲染错误: {str(e)}")
        finally:
            qp.end()

    def _draw_click_effect(self, qp):
        if self.last_click_pos and self.click_animation > 0:
            radius = 25 * self.click_animation
            alpha = int(150 * self.click_animation)
            
            qp.resetTransform()
            gradient = QRadialGradient(self.last_click_pos, radius)
            gradient.setColorAt(0, QColor(255, 200, 0, alpha))
            gradient.setColorAt(1, QColor(255, 100, 0, 0))
            
            qp.setPen(Qt.NoPen)
            qp.setBrush(QBrush(gradient))
            qp.drawEllipse(self.last_click_pos, radius, radius)

    def _draw_ecosystem(self, qp, visible_rect):
        try:
            # 优化背景绘制
            qp.fillRect(visible_rect, QColor(15, 15, 25))
            
            # 危险边界
            qp.setPen(QPen(QColor(255, 50, 50, 150), 15))
            qp.setBrush(Qt.NoBrush)
            qp.drawRect(8, 8, self.width() - 16, self.height() - 16)

            sim = self.parent.simulator
            
            # 优化果实绘制
            if sim.fruit_positions.shape[1] > 0:
                fpos = sim.fruit_positions.cpu().numpy()
                fvals = sim.fruit_values.cpu().numpy()
                
                # 空间过滤
                mask = (fpos[0] >= 0) & (fpos[0] <= 1.0) & \
                       (fpos[1] >= 0) & (fpos[1] <= 1.0)
                fpos = fpos[:, mask]
                fvals = fvals[mask]
                
                for i in range(min(fpos.shape[1], sim.MAX_DRAW_ENTITIES)):
                    x = fpos[0, i] * self.width()
                    y = fpos[1, i] * self.height()
                    size = int(np.clip(fvals[i]/50, 8, 20))
                    if visible_rect.contains(QPointF(x, y)):
                        gradient = QRadialGradient(x, y, size)
                        gradient.setColorAt(0, QColor(255, 230, 80))
                        gradient.setColorAt(1, QColor(255, 160, 0, 100))
                        qp.setBrush(QBrush(gradient))
                        qp.drawEllipse(QPointF(x, y), size, size)

            # 优化生物绘制
            pos = sim.positions.cpu().numpy()
            scores = sim.scores.cpu().numpy()
            energy = sim.energy.cpu().numpy()
            gens = sim.generations.cpu().numpy()
            alive = sim.alive.cpu().numpy()
            categories = sim.category.cpu().numpy()
            
            # 空间过滤
            mask = (pos[0] >= 0) & (pos[0] <= 1.0) & \
                   (pos[1] >= 0) & (pos[1] <= 1.0) & \
                   alive
            indices = np.where(mask)[0][:sim.MAX_DRAW_ENTITIES]
            
            for i in indices:
                x = pos[0, i] * self.width()
                y = pos[1, i] * self.height()
                # 调整大小计算公式，使成长更明显
                base_size = 8 + gens[i] * 0.5  # 基础大小随世代增加
                score_size = scores[i] ** 0.6   # 调整指数使初期变化更明显
                size = int(base_size + score_size)
                
                color = QColor(*sim.CATEGORY_TRAITS[categories[i]]['color'])
                
                # 战斗光环
                if categories[i] == 1 and energy[i] > 60:
                    qp.setPen(QPen(QColor(255, 0, 0, 80), 4))
                    qp.setBrush(Qt.NoBrush)
                    qp.drawEllipse(QPointF(x, y), size*1.3, size*1.3)
                
                # 能量警示
                border_width = 3 + int((100 - energy[i])/15)
                border_color = QColor(255, 80, 80, 220) if energy[i] < 50 else color.darker(160)
                
                qp.setPen(QPen(border_color, border_width))
                qp.setBrush(color)
                qp.drawEllipse(QPointF(x, y), size, size)
        except Exception as e:
            warnings.warn(f"绘制错误: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.simulator = SafeSimulator("cuda" if torch.cuda.is_available() else "cpu")
        self._init_ui()
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._safe_simulate_step)
        self.sim_timer.start(1000 // 60)

    def _init_ui(self):
        self.setWindowTitle("终极生存模拟器 v7.1")
        self.setGeometry(100, 100, 1400, 800)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧可视化区
        left_frame = QFrame()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.visualizer = SafeVisualizer(self)
        left_layout.addWidget(self.visualizer)

        # 状态栏
        self.status_label = QLabel()
        self.status_label.setFont(QFont("Consolas", 10))
        self.status_label.setStyleSheet("color: white; background: #333; padding: 5px;")
        left_layout.addWidget(self.status_label)

        left_frame.setLayout(left_layout)
        main_layout.addWidget(left_frame, stretch=7)

        # 右侧控制面板
        right_frame = QFrame()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)

        # 控制组
        control_group = QGroupBox("模拟控制")
        control_layout = QVBoxLayout()

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self._toggle_pause)

        self.population_spin = QSpinBox()
        self.population_spin.setRange(1000, 20000)
        self.population_spin.setValue(self.simulator.population)
        self.population_spin.valueChanged.connect(self._update_population)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 240)
        self.speed_slider.setValue(60)
        self.speed_slider.valueChanged.connect(self._update_speed)

        self.device_btn = QPushButton("切换设备")
        self.device_btn.clicked.connect(self._toggle_device)

        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(QLabel("数量:"))
        control_layout.addWidget(self.population_spin)
        control_layout.addWidget(QLabel("速度:"))
        control_layout.addWidget(self.speed_slider)
        control_layout.addWidget(self.device_btn)
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        # 统计组
        stats_group = QGroupBox("实时统计")
        stats_layout = QVBoxLayout()

        self.stats_labels = {
            'population': QLabel("生物数量: -"),
            'alive': QLabel("存活数量: -"),
            'fruits': QLabel("果实数量: -"),
            'max_score': QLabel("最高积分: -"),
            'avg_generation': QLabel("平均世代: -"),
            'boundary_deaths': QLabel("边界死亡: 0"),
            'category_0': QLabel("标准型: 0"),
            'category_1': QLabel("掠食型: 0"),
            'category_2': QLabel("繁殖型: 0")
        }

        for label in self.stats_labels.values():
            label.setFont(QFont("Consolas", 9))
            label.setStyleSheet("color: #DDD;")
            stats_layout.addWidget(label)

        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # 操作提示
        tips_group = QGroupBox("操作提示")
        tips_layout = QVBoxLayout()
        tips = [
            "左键点击: 放置果实",
            "中键拖动: 移动视图",
            "滚轮: 缩放视图",
            "空格键: 暂停/继续"
        ]
        for tip in tips:
            label = QLabel(tip)
            label.setStyleSheet("color: #AAA;")
            tips_layout.addWidget(label)
        
        tips_group.setLayout(tips_layout)
        right_layout.addWidget(tips_group)

        right_frame.setLayout(right_layout)
        main_layout.addWidget(right_frame, stretch=3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self._update_status()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._toggle_pause()

    def _safe_simulate_step(self):
        try:
            if not self.simulator.paused:
                self.simulator.update_ecosystem(1 / 60)
                self._update_status()
            self.visualizer.update()
        except Exception as e:
            warnings.warn(f"模拟错误: {str(e)}")
            self.sim_timer.stop()

    def _toggle_pause(self):
        self.simulator.paused = not self.simulator.paused
        self.pause_btn.setText("继续" if self.simulator.paused else "暂停")
        self._update_status()

    def _update_population(self, value):
        try:
            self.simulator.population = value
            self.simulator.reset_ecosystem()
            self._update_status()
        except Exception as e:
            warnings.warn(f"数量更新失败: {str(e)}")

    def _update_speed(self, value):
        self.sim_timer.setInterval(1000 // value)

    def _toggle_device(self):
        try:
            new_device = "cpu" if self.simulator.device.type == "cuda" else "cuda"

            if new_device == "cuda" and not torch.cuda.is_available():
                QMessageBox.warning(self, "警告", "CUDA设备不可用")
                return

            was_paused = self.simulator.paused
            self.simulator.paused = True

            new_population = self.simulator.population
            self.simulator = SafeSimulator(new_device, new_population)
            self.simulator.paused = was_paused
            
            self.population_spin.blockSignals(True)
            self.population_spin.setValue(new_population)
            self.population_spin.blockSignals(False)
            self._update_status()
        except Exception as e:
            QMessageBox.critical(self, "设备切换错误", str(e))

    def _update_status(self):
        try:
            alive_count = self.simulator.alive.sum().item()
            avg_gen = self.simulator.generations.float().mean().item() if alive_count > 0 else 0.0
            category_counts = [
                (self.simulator.category == i).sum().item() for i in range(3)
            ]

            stats = {
                'population': f"生物数量: {self.simulator.population}",
                'alive': f"存活数量: {alive_count}",
                'fruits': f"果实数量: {self.simulator.fruit_positions.shape[1]}",
                'max_score': f"最高积分: {self.simulator.scores.max().item():.1f}" if alive_count > 0 else "最高积分: -",
                'avg_generation': f"平均世代: {avg_gen:.1f}",
                'boundary_deaths': f"边界死亡: {self.simulator.boundary_deaths}",
                'category_0': f"标准型: {category_counts[0]}",
                'category_1': f"掠食型: {category_counts[1]}",
                'category_2': f"繁殖型: {category_counts[2]}"
            }

            for key, label in self.stats_labels.items():
                label.setText(stats[key])

            # 状态栏
            device_status = f"设备: {self.simulator.device.type.upper()}"
            speed_status = f"速度: {self.speed_slider.value()} FPS"
            pause_status = "| 已暂停" if self.simulator.paused else ""
            self.status_label.setText(f"{device_status} | {speed_status} {pause_status}")
        except Exception as e:
            warnings.warn(f"状态更新失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        warnings.warn(f"应用错误: {str(e)}")
