# -*- coding: utf-8 -*-
"""
终极生存模拟器 v6.5
完整修复版
修复内容：
1. 修复所有括号匹配错误
2. 统一代码格式规范
3. 增强异常处理
"""

import sys
import os
import warnings
import numpy as np
import torch
from PyQt5.QtCore import QTimer, Qt, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSlider,
                             QSpinBox, QGroupBox, QFrame)

# 环境配置
warnings.filterwarnings("ignore")
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.*.debug=false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class SafeSimulator:
    MAX_DRAW_ENTITIES = 10000
    MAX_FRUITS = 5000
    INITIAL_ENERGY = 100.0
    BASE_ENERGY_DECAY = 3.0
    CATEGORY_TRAITS = {
        0: {'color': (0, 160, 255), 'speed': 1.0, 'decay': 3.0, 'gain': 1.0},
        1: {'color': (255, 80, 0), 'speed': 1.3, 'decay': 4.0, 'gain': 0.8},
        2: {'color': (50, 255, 100), 'speed': 1.0, 'decay': 2.5, 'gain': 1.5}
    }

    def __init__(self, device_type, population=10000):
        self.device = self._init_device(device_type)
        self.population = min(population, 50000 if self.device.type == 'cuda' else 10000)
        self.world_size = 1000
        self.paused = False
        self.boundary_deaths = 0
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
            active = self.alive.clone()
            self.age[active] += 1

            # 死亡检测（修复括号匹配）
            death_conditions = [
                (self.age[active] > self.max_age[active], False),
                ((self.positions[0] < 0.01) | (self.positions[0] > 0.99) |
                 (self.positions[1] < 0.01) | (self.positions[1] > 0.99), True),
                (self.energy[active] <= 0, False)
            ]

            for mask, is_boundary in death_conditions:
                if mask.any():
                    self._process_death(mask, is_boundary)

            # 能量消耗
            if active.any():
                traits = torch.tensor(
                    [self.CATEGORY_TRAITS[c.item()]['decay'] for c in self.category[active]],
                    device=self.device
                )
                self.energy[active] -= torch.rand_like(self.energy[active]) * (traits + self.generations[active] * 0.1) * dt
                self.energy[active] = torch.clamp(self.energy[active], 0.0, 200.0)

            # 生物行为
            if active.any():
                self._update_movement(dt, active)
                self._update_feeding(active)
                self._update_reproduction(active)
        except Exception as e:
            warnings.warn(f"更新失败: {str(e)}")

    def _process_death(self, dead_mask, boundary_death):
        try:
            dead_pos = self.positions[:, dead_mask]
            self.spawn_fruit(dead_pos, self.scores[dead_mask] * (0.9 if boundary_death else 0.7))
            
            self.alive[dead_mask] = False
            self.scores[dead_mask] = 0
            self.generations[dead_mask] = 0
            self.age[dead_mask] = 0
            
            if boundary_death:
                self.boundary_deaths += dead_mask.sum().item()
        except Exception as e:
            warnings.warn(f"死亡处理失败: {str(e)}")

    def _update_movement(self, dt, active):
        try:
            speeds = torch.tensor(
                [self.CATEGORY_TRAITS[c.item()]['speed'] for c in self.category[active]],
                device=self.device
            ).unsqueeze(0)

            if self.fruit_positions.shape[1] > 0:
                pos = self.positions[:, active].unsqueeze(2)
                fruits = self.fruit_positions.unsqueeze(1)
                closest = torch.argmin(torch.norm(fruits - pos, dim=0), dim=1)
                move_dir = (fruits - pos)[:, torch.arange(pos.shape[1]), closest]
                self.velocities[:, active] += move_dir * 0.015 * dt * speeds

            self.velocities[:, active] = torch.clamp(
                self.velocities[:, active] + torch.randn_like(self.velocities[:, active]) * 0.02,
                -0.5, 0.5
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

            pos = self.positions[:, active].unsqueeze(2)
            fruits = self.fruit_positions.unsqueeze(1)
            in_range = torch.norm(pos - fruits, dim=0) < 0.015

            if in_range.any():
                eaters, fruits_eaten = torch.where(in_range)
                unique_eaters, counts = torch.unique(eaters, return_counts=True)

                gains = torch.tensor(
                    [self.CATEGORY_TRAITS[c.item()]['gain'] for c in self.category[unique_eaters]],
                    device=self.device
                )
                energy_gain = (self.fruit_values[fruits_eaten] / counts.float() * gains)[unique_eaters]

                self.energy[unique_eaters] += energy_gain * 5
                self.scores[unique_eaters] += energy_gain * 2

                remaining_fruits = ~torch.any(in_range, dim=0)
                self.fruit_positions = self.fruit_positions[:, remaining_fruits]
                self.fruit_values = self.fruit_values[remaining_fruits]
        except Exception as e:
            warnings.warn(f"进食更新失败: {str(e)}")

    def _update_reproduction(self, active):
        try:
            repro_mask = (self.energy > 50) & (self.age > 20) & active
            if not repro_mask.any():
                return

            parents = torch.where(repro_mask)[0]
            child_count = parents.size(0)
            
            # 遗传变异
            parent_cat = self.category[parents]
            child_cat = torch.where(
                torch.rand(child_count, device=self.device) < 0.05,
                torch.randint(0, 3, (child_count,), device=self.device),
                parent_cat
            )

            # 后代生成
            child_pos = torch.clamp(
                self.positions[:, parents] + torch.randn((2, child_count), device=self.device) * 0.02,
                0.05, 0.95
            )
            child_vel = self.velocities[:, parents] * (0.8 + torch.rand((1, child_count), device=self.device) * 0.4)
            
            # 种群扩展
            self.positions = torch.cat([self.positions, child_pos], dim=1)
            self.velocities = torch.cat([self.velocities, child_vel], dim=1)
            self.energy = torch.cat([self.energy, torch.full((child_count,), 50.0, device=self.device)])
            self.scores = torch.cat([self.scores, torch.zeros(child_count, device=self.device)])
            self.generations = torch.cat([self.generations, self.generations[parents] + 1])
            self.age = torch.cat([self.age, torch.zeros(child_count, device=self.device)])
            self.max_age = torch.cat([self.max_age, torch.randint(50, 200, (child_count,), device=self.device)])
            self.category = torch.cat([self.category, child_cat])
            self.alive = torch.cat([self.alive, torch.ones(child_count, dtype=torch.bool, device=self.device)])

            self.energy[parents] -= 30
            self.population += child_count
        except Exception as e:
            warnings.warn(f"繁殖更新失败: {str(e)}")

class SafeVisualizer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedSize(self.parent.simulator.world_size, self.parent.simulator.world_size)
        self.last_click_pos = None
        self.click_animation = 0.0
        self.setMouseTracking(True)

    def paintEvent(self, event):
        qp = QPainter(self)
        try:
            self._draw_ecosystem(qp)
            self._draw_click_effect(qp)
        except Exception as e:
            warnings.warn(f"渲染错误: {str(e)}")
        finally:
            qp.end()

    def mousePressEvent(self, event):
        try:
            if self.parent.simulator.paused or event.button() != Qt.LeftButton:
                return

            pos = event.pos()
            norm_pos = torch.tensor([
                pos.x() / self.width(),
                pos.y() / self.height()
            ], device=self.parent.simulator.device)
            
            self.parent.simulator.spawn_fruit(
                norm_pos,
                torch.tensor([np.random.uniform(50, 100)], device=self.parent.simulator.device)
            )
            
            self.last_click_pos = pos
            self.click_animation = 1.0
            QTimer.singleShot(100, self._update_animation)
        except Exception as e:
            warnings.warn(f"点击处理错误: {str(e)}")

    def _update_animation(self):
        self.click_animation -= 0.1
        if self.click_animation > 0:
            self.update()
            QTimer.singleShot(50, self._update_animation)
        else:
            self.last_click_pos = None

    def _draw_click_effect(self, qp):
        if self.last_click_pos and self.click_animation > 0:
            radius = 15 * self.click_animation
            alpha = int(100 * self.click_animation)
            
            gradient = QRadialGradient(self.last_click_pos, radius)
            gradient.setColorAt(0, QColor(255, 200, 0, alpha))
            gradient.setColorAt(1, QColor(255, 100, 0, 0))
            
            qp.setPen(Qt.NoPen)
            qp.setBrush(QBrush(gradient))
            qp.drawEllipse(self.last_click_pos, radius, radius)

    def _draw_ecosystem(self, qp):
        try:
            # 背景
            qp.fillRect(0, 0, self.width(), self.height(), QColor(25, 25, 35))
            
            # 边界
            qp.setPen(QPen(QColor(255, 50, 50, 100), 10))
            qp.setBrush(Qt.NoBrush)
            qp.drawRect(5, 5, self.width() - 10, self.height() - 10)

            sim = self.parent.simulator
            
            # 果实
            if sim.fruit_positions.shape[1] > 0:
                fpos = sim.fruit_positions.cpu().numpy() * self.width()
                fvals = sim.fruit_values.cpu().numpy()
                for i in range(min(fpos.shape[1], sim.MAX_DRAW_ENTITIES)):
                    x = int(fpos[0, i])
                    y = int(fpos[1, i])
                    size = int(np.clip(fvals[i]/30, 5, 15))
                    gradient = QRadialGradient(x, y, size)
                    gradient.setColorAt(0, QColor(255, 223, 0))
                    gradient.setColorAt(1, QColor(255, 153, 0, 50))
                    qp.setBrush(QBrush(gradient))
                    qp.drawEllipse(QPointF(x, y), size, size)

            # 生物
            pos = sim.positions.cpu().numpy() * self.width()
            scores = sim.scores.cpu().numpy()
            gens = sim.generations.cpu().numpy()
            alive = sim.alive.cpu().numpy()
            categories = sim.category.cpu().numpy()
            
            for i in range(min(alive.shape[0], sim.MAX_DRAW_ENTITIES)):
                if alive[i]:
                    x = int(pos[0, i])
                    y = int(pos[1, i])
                    size = int(3 + scores[i]**0.6/4 + gens[i]*0.4)
                    color = QColor(*sim.CATEGORY_TRAITS[categories[i]]['color'])
                    
                    qp.setPen(QPen(color.darker(), 1))
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
        self.setWindowTitle("终极生存模拟器 v6.5")
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
        self.population_spin.setRange(1000, 50000)
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

        right_frame.setLayout(right_layout)
        main_layout.addWidget(right_frame, stretch=3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self._update_status()

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
