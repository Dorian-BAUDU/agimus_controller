import time
import example_robot_data
import numpy as np
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.plots import MPCPlots
from agimus_controller.utils.build_models import get_robot_model, get_collision_model
from agimus_controller.utils.path_finder import get_project_root
from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP

if __name__ == "__main__":
    pandawrapper = PandaWrapper(auto_col=True)
    project_root_path = get_project_root()
    urdf_path = str(project_root_path / "urdf/robot.urdf")
    srdf_path = str(project_root_path / "srdf/demo.srdf")
    collision_params_path = str(project_root_path / "config/param.yaml")
    robot = example_robot_data.load("panda")
    rmodel = get_robot_model(robot, urdf_path, srdf_path)
    cmodel = get_collision_model(rmodel, urdf_path, collision_params_path)
    rmodel, cmodel, vmodel = pandawrapper.create_robot()
    ee_frame_name = pandawrapper.get_ee_frame_name()
    hpp_interface = HppInterface()
    q_init, q_goal = hpp_interface.get_panda_q_init_q_goal()
    hpp_interface.set_panda_planning(q_init, q_goal)
    ps, viewer = hpp_interface.get_problem_solver_and_viewer()
    x_plan, a_plan, whole_traj_T = hpp_interface.get_hpp_x_a_planning(
        1e-2, 7, ps.client.problem.getPath(ps.numberPaths() - 1)
    )
    armature = np.zeros(rmodel.nq)
    ocp = OCPCrocoHPP(rmodel, cmodel, use_constraints=False, armature=armature)

    mpc = MPC(ocp, x_plan, a_plan, rmodel, cmodel)
    start = time.time()
    mpc.ocp.set_weights(10**4, 1, 10**-3, 0)
    mpc.simulate_mpc(T=100, save_predictions=False)
    end = time.time()
    u_plan = mpc.ocp.get_u_plan(x_plan, a_plan)
    mpc_plots = MPCPlots(
        croco_xs=mpc.croco_xs,
        croco_us=mpc.croco_us,
        whole_x_plan=x_plan,
        whole_u_plan=u_plan,
        rmodel=rmodel,
        DT=mpc.ocp.DT,
        ee_frame_name=ee_frame_name,
        viewer=viewer,
    )
