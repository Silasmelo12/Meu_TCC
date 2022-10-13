import meshio as mio
import matplotlib.pyplot as plt
import h5py
from fenics import *
import dolfin as df
from tqdm import tqdm

def menu_fluido():
    while True:
        print("informe o fluido\n\n"
              "1 - oleo\n")
        fluido_type = int(input())
        if fluido_type == 1:
            return fluido_type

def carregar_malha(caminho):
    mesh_from_file = mio.read(caminho)
    return mesh_from_file


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = mio.Mesh(points=mesh.points, 
    cells={cell_type: cells}, 
                        cell_data={"name_to_read": [cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


def mvc_mf(mesh_from_file):
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    mio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle",
    prune_z=True)
    mio.write("mesh.xdmf", triangle_mesh)

    mesh = Mesh()
    
    with XDMFFile("mesh.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)

    with XDMFFile("facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    mf = MeshFunction("size_t", mesh, mvc)
    plot(mesh)
    plt.show()
    return mesh, mvc, mf


def dados_entrada(type_fluido):
    dados_problema = {'diametro_saida':40,
                      'Tempo': 10,
                      'grav':-9.85,
                      'num_steps': 40000}
    Reynolds = 50
    dados_problema['velocidade_fluido_inflowX'] = 0
    dados_problema['velocidade_fluido_inflowY'] =
    -1000*(Reynolds*0.0381)/(900*0.04)
    dados_problema['dt'] =
    int(dados_problema['Tempo'])/int(dados_problema['num_steps'])
    
    oleo = {
        'nome': "oleo",
        'viscosidade': 38.1,
        'densidade': 0.000900
    }

    dados_fluidos = [oleo]
    Reynolds = (Reynolds*0.0381)/(900*40)
    return dados_fluidos[type_fluido - 1],dados_problema,Reynolds


def delete_pasta_simulacao():
    import shutil

    try:
        shutil.rmtree('TesteIPCS1')
        shutil.rmtree('navier_stokes_cylinder')
    except OSError as e:
        print(f"Error:{ e.strerror}")


def modelo(mesh, dados_problema,dados_fluidos, mf):
    V = VectorFunctionSpace(mesh, 'P', 2)  
    Q = FunctionSpace(mesh, 'P', 1)  

    inflow_profile = ('0' +
    str(dados_problema['velocidade_fluido_inflowX']),
    '0' + str(dados_problema['velocidade_fluido_inflowY']))

    bcu_inflow = DirichletBC(V, Expression(inflow_profile,
    degree=2), mf, 1)
    bcp_outflow = DirichletBC(Q, Constant(0), mf, 2)
    bcp_paredes = DirichletBC(V, Constant((0, 0)), mf, 3)
    bcu = [bcu_inflow, bcp_paredes]  
    bcp = [bcp_outflow]  

    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    u0 = Function(V)
    u_ = Function(V)
    p_n = Function(Q)
    p_ = Function(Q)

    U = 0.5 * (u0 + u)
    n = FacetNormal(mesh)
    f = Constant((0, dados_problema['grav']*
    dados_fluidos['densidade']))
    k = Constant(dados_problema['dt'])

    viscosidadeCinematica =
    Constant(dados_fluidos['viscosidade'])
    densidade = Constant(dados_fluidos['densidade'])
    beta = 1

    def epsilon(u):
        return (1 / 2) * (nabla_grad(u) + nabla_grad(u).T)

    def sigma(u, p, viscosidadeCinematica):
        return 2 * viscosidadeCinematica * epsilon(u) - p *
        Identity(len(u))

    # tentativa de velocidade
    F1 = (1.0 / k) * inner(u - u0, v) * df.dx \
         + inner(grad(u0) * u0, v) * df.dx \
         + inner(sigma(U, p_n, viscosidadeCinematica),
         epsilon(v)) * df.dx \
         + inner(p_n * n, v) * df.ds \
         - beta * viscosidadeCinematica * inner(grad(U).T * n, v)
         * df.ds \
         - viscosidadeCinematica * inner(f, v) * df.dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Correcao da pressao
    a2 = inner(grad(p), grad(q)) * df.dx
    L2 = inner(grad(p_n), grad(q)) * df.dx \
         - (1.0 / k) * div(u_) * q * df.dx

    # Correcao da velocidade
    a3 = inner(u, v) * df.dx
    L3 = inner(u_, v) * df.dx \
         - k * inner(grad(p_ - p_n), v) * df.dx

    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    modelo = {'A1': A1,
              'A2': A2,
              'A3': A3,
              'bcu': bcu,
              'bcp': bcp,
              'dados_problema':dados_problema,
              'L1': L1,
              'L2': L2,
              'L3': L3,
              'u_': u_,
              'p_': p_,
              'u0': u0,
              'p_n': p_n}
    return modelo


def solucao(modelo):
    passos = 0
    t = 0
    for n in tqdm(range(modelo['dados_problema']['num_steps'])):
        
        [bc.apply(modelo['A1']) for bc in modelo['bcu']]
        [bc.apply(modelo['A2']) for bc in modelo['bcp']]

        # Tentativa de velocidade
        b1 = assemble(modelo['L1'])
        [bc.apply(modelo['A1'], b1) for bc in modelo['bcu']]
        solve(modelo['A1'], modelo['u_'].vector(), b1,
        'bicgstab', 'hypre_amg')

        # Correcao de Pressao
        b2 = assemble(modelo['L2'])
        [bc.apply(b2) for bc in modelo['bcp']]
        solve(modelo['A2'], modelo['p_'].vector(), b2,
        'bicgstab', 'hypre_amg')

        # Correcao da velocidade
        b3 = assemble(modelo['L3'])
        solve(modelo['A3'], modelo['u_'].vector(), b3, 'cg',
        'sor')

        if n % 200 == 0:
            pvd_file = File('TesteIPCS1/
            velocidade-{0}.pvd'.format(passos))
            pvd_file << modelo['u_']
            pvd_file = File('TesteIPCS1/
            pressao-{0}.pvd'.format(passos))
            pvd_file << modelo['p_']
            passos = passos + 1

        modelo['u0'].assign(modelo['u_'])
        modelo['p_n'].assign(modelo['p_'])
        t = t + modelo['dados_problema']['dt']
    return modelo

if __name__ == "__main__":
    fluido_type = menu_fluido()
    caminho = "lisav2.msh"
    mesh_from_file = carregar_malha(caminho)
    mesh, mvc, mf = mvc_mf(mesh_from_file)
    dados_fluidos, dados_problema, Reynolds  =
    dados_entrada(fluido_type)
    modelo = modelo(mesh, dados_problema,dados_fluidos, mf)
    resultado = solucao(modelo)

name = 100
caminho_lisa   =
str("liso/standoff/"+str(name)+".csv")
caminho_rugoso =
str("rugoso/standoff/"+str(name)+".csv")
df_rugoso = pd.read_csv(caminho_rugoso)
df_lisa = pd.read_csv(caminho_lisa)

df_lisa = df_lisa.dropna().reset_index(drop=True)
df_rugoso = df_rugoso.dropna().reset_index(drop=True)

df_lisa['f_24:0'] = df_lisa['f_24:0']/1000
df_lisa['f_24:1'] = df_lisa['f_24:1']/1000
df_lisa['Points:0'] = df_lisa['Points:0']/1000
df_rugoso['f_24:0']=df_rugoso['f_24:0']/1000
df_rugoso['f_24:1']=df_rugoso['f_24:1']/1000
df_rugoso['Points:0']=df_rugoso['Points:0']/1000

vx_lisa = df_lisa['f_24:0']
vx_rugoso = df_rugoso['f_24:0']
vy_lisa = df_lisa['f_24:1']
vy_rugoso = df_rugoso['f_24:1']
v_lisa = np.sqrt(vx_lisa**2 + vy_lisa**2).values
v_rugoso = np.sqrt(vx_rugoso**2 + vy_rugoso**2).values
v_lisa = vy_lisa
v_rugoso = vy_rugoso

x_lisa = df_lisa['Points:0'].values
x_rugoso = df_rugoso['Points:0'].values

nao_nulo_esq_lisa = np.argwhere((x_lisa < 100/1000)).flatten()
nao_nulo_esq_rugoso = np.argwhere((x_rugoso < 100/1000)).flatten()
nao_nulo_dir_lisa = np.argwhere((x_lisa > 100/1000)).flatten()
nao_nulo_dir_rugoso = np.argwhere((x_rugoso > 100/1000)).flatten()

E_lisa = x_lisa[nao_nulo_esq_lisa]; 
E_rugoso = x_rugoso[nao_nulo_esq_rugoso]; 
D_lisa = x_lisa[nao_nulo_dir_lisa];
D_rugoso= x_rugoso[nao_nulo_dir_rugoso];
h_e_lisa = np.max(E_lisa) - np.min(E_lisa)
h_e_rugoso = np.max(E_rugoso) - np.min(E_rugoso)
h_d_lisa = np.max(D_lisa) - np.min(D_lisa)
h_d_rugoso = np.max(D_rugoso) - np.min(D_rugoso)

v_e_lisa = v_lisa[nao_nulo_esq_lisa]; 
v_e_rugoso = v_rugoso[nao_nulo_esq_rugoso]; 
v_d_lisa = v_lisa[nao_nulo_dir_lisa]
v_d_rugoso = v_rugoso[nao_nulo_dir_rugoso]
Q_e_lisa = trapezoid(v_e_lisa)
Q_e_rugoso = trapezoid(v_e_rugoso)

Q_d_lisa = trapezoid(v_d_lisa)
Q_d_rugoso = trapezoid(v_d_rugoso)

deltax = 0.1
delta_e = np.min(E_lisa)+(h_e_lisa*deltax)
delta_d = np.max(D_lisa)-(h_d_lisa*deltax)

limite_zona_limpa_e = (delta_e,np.max(E_lisa))
limite_zona_limpa_d = (np.min(D_lisa),delta_d)

regiao_limpa_e_lisa   = np.argwhere(((x_lisa < 
limite_zona_limpa_e[1])&(x_lisa >
limite_zona_limpa_e[0]))).
flatten()
regiao_limpa_d_lisa   = np.argwhere(((x_lisa <
limite_zona_limpa_d[1])&(x_lisa > limite_zona_limpa_d[0]))).
flatten()

regiao_limpa_e_rugoso = np.argwhere(((x_rugoso < 
limite_zona_limpa_e[1])&
(x_rugoso > limite_zona_limpa_e[0]))).flatten()
regiao_limpa_d_rugoso = np.argwhere(((x_rugoso < 
limite_zona_limpa_d[1])&(x_rugoso > limite_zona_limpa_d[0]))).flatten()

Qlimpo_e_lisa   = trapezoid(v_lisa[regiao_limpa_e_lisa])
Qlimpo_e_rugoso = trapezoid(v_rugoso[regiao_limpa_e_rugoso])

Qlimpo_d_lisa   = trapezoid(v_lisa[regiao_limpa_d_lisa])
Qlimpo_d_rugoso = trapezoid(v_rugoso[regiao_limpa_d_rugoso])

Q_A_E = Q_e_lisa
Q_A90_E = Qlimpo_e_lisa
Q_B_E = Q_e_rugoso
Q_B90_E = Qlimpo_e_rugoso
EtaE90 = Q_B90_E/Q_A90_E
EtaE = Q_B_E/Q_A_E

Q_A_D = Q_d_lisa
Q_A90_D = Qlimpo_d_lisa
Q_B_D = Q_d_rugoso
Q_B90_D = Qlimpo_d_rugoso
EtaD90 = Q_B90_D/Q_A90_D
EtaD = Q_B_D/Q_A_D
