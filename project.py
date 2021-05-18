import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

ae = 149597870700
le = 63241.1 * ae
parcek = ae * 206265
inner_eccentricity = 0.7
outer_eccentricity = 0.9
angular_offset = 0.0001 / le # от 0.0003/масштаб до 0.0007/масштаб
core_radius = 1000 * parcek
galaxy_radius = 50000 * le

vxc=0
vyc=0
xc=0
yc=0
N = 7000

G = 6.67 * 10 ** (-11)
m_g = 9.399742662685108e+39

gx0_one = 1
gy0_one = - 30000 * le
gv_x0_one = 147000
gv_y0_one = 0

print(gv_x0_one)

gx0_two = 1
gy0_two = 30000 * le
gv_x0_two = - 147000
gv_y0_two = 0

print(gv_x0_two)

s0 = (gx0_one, gv_x0_one, gy0_one, gv_y0_one,
      gx0_two, gv_x0_two, gy0_two, gv_y0_two)

frames = 1400
t = np.linspace(0, 5.3072e+17, frames)

"""
Функция распределения частиц в галактике
inner_eccentricity - Эксцентричность внутреннего эллипса
outer_eccentricity - Эксцентриситет внешнего эллипса
angular_offset - Угловое смещение на парсек
core_radius - Внутренний радиус ядра
galaxy_radius - Радиус галактики
N - Количество звезд
vxc - "х" компонента скорости центра
vyc - "у" компонента скорости центра
G - гравитационная постоянная
m_g - масса галактики
gx0_one - положение объекта 1 на оси х
gy0_one - положение объекта 1 на оси у
gv_x0_one - начальная скорость объекта 1 по оси х
gv_y0_one - начальная скорость объекта 1 по оси у
gx0_two - положение объекта 2 на оси х
gy0_two - положение объекта 2 на оси у
gv_x0_two - начальная скорость объекта 2 по оси х
gv_y0_two - начальная скорость объекта 2 по оси у
xc - начальные координаты центра
yc - начальные координаты центра
frames - количество кадров
"""

distant_radius = galaxy_radius * 2 # Радиус, после которого все волны
                                   # плотности должны иметь округлую форму.
# Создания массивов данных для частиц
theta = np.ndarray(shape=(N))
angle = np.ndarray(shape=(N))
m_a = np.ndarray(shape=(N))
m_b = np.ndarray(shape=(N))
coordinate = np.ndarray(shape=(N, 2))
velocity = np.ndarray(shape=(N, 2))

# Функция рассчитывает эксцентриситет
def eccentricity(r):

    if r < core_radius:
        return 1 + (r / core_radius) * (inner_eccentricity-1)

    elif r > core_radius and r <= galaxy_radius:
        a = galaxy_radius - core_radius
        b = outer_eccentricity - inner_eccentricity
        return inner_eccentricity + (r - core_radius) / a * b

    elif r > galaxy_radius and r < distant_radius:
        a = distant_radius - galaxy_radius
        b = 1 - outer_eccentricity
        return outer_eccentricity + (r - galaxy_radius) / a * b

    else:
        return 1

def galaxy_func(s, t):

  (gx_one, gv_x_one, gy_one, gv_y_one,
  gx_two, gv_x_two, gy_two, gv_y_two) = s

  dxdt_one = gv_x_one
  dv_xdt_one = - G * m_g * gx_one / (gx_one ** 2 + gy_one ** 2) ** 1.5
  dydt_one = gv_y_one
  dv_ydt_one = - G * m_g * gy_one / (gx_one ** 2 + gy_one ** 2) ** 1.5

  dxdt_two = gv_x_two
  dv_xdt_two = - G * m_g * gx_two / (gx_two ** 2 + gy_two ** 2) ** 1.5
  dydt_two = gv_y_two
  dv_ydt_two = - G * m_g * gy_two / (gx_two ** 2 + gy_two ** 2) ** 1.5

  return (dxdt_one, dv_xdt_one, dydt_one, dv_ydt_one,
          dxdt_two, dv_xdt_two, dydt_two, dv_ydt_two)

def solve_func(j):
  sol = odeint(galaxy_func, s0, t)
  g_x_one = sol[j, 0]
  g_y_one = sol[j, 2]

  g_x_two = sol[j, 4]
  g_y_two = sol[j, 6]

  return (g_x_one, g_y_one), (g_x_two, g_y_two)

# Инициализация  звёзд
X = np.random.uniform(-galaxy_radius, galaxy_radius, N)
Y = np.random.uniform(-galaxy_radius, galaxy_radius, N)
R = np.sqrt(X*X+Y*Y)
m_a = R + 1000
angle = R * angular_offset
theta = np.random.uniform(0, 360, N)
m_b = np.ndarray(shape=(N))
for i in range(N):
    m_b[i] = R[i] * eccentricity(R[i])

# Анимирование
fig, ax  = plt.subplots(figsize=(10,10))
ax.set_xlim(-100000*le, 100000*le)
ax.set_ylim(-100000*le, 100000*le)
ax.set_facecolor('black')
stars, = plt.plot([],[],'.', ms='1', color='white')
point_one, = plt.plot([],[],'.', ms='10', color='orange')
point_two, = plt.plot([],[],'.', ms='10', color='orange')


coor_x = np.ndarray(shape=(N))
coor_y = np.ndarray(shape=(N))

def func(timestep=1):
    for i in range(0, N, 1):
        theta[i] += 0.05 * timestep
        alpha = np.ndarray(shape=(N))
        alpha[i] = theta[i] * np.pi / 180.0
        x = xc + m_a[i]*np.cos(alpha[i])*np.cos(-angle[i]) - m_b[i]*np.sin(alpha[i])*np.sin(-angle[i])
        y = yc + m_a[i]*np.cos(alpha[i])*np.sin(-angle[i]) + m_b[i]*np.sin(alpha[i])*np.cos(-angle[i])
        coor_x[i] = x
        coor_y[i] = y
    return coor_x, coor_y


def update(j):
    stars.set_data(func(timestep=j)[0],func(timestep=j)[1])
    point_one.set_data(solve_func(j)[0])
    point_two.set_data(solve_func(j)[1])

ani = animation.FuncAnimation(fig,
                              update,
                              frames=frames,
                              interval=30)



plt.show()
