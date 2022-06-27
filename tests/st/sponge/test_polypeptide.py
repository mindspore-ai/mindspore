from tests.st.sponge.simulation_np import Simulation
import pytest
import mindspore.context as context


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sponge_polypeptide():
    """
    Feature: Test the polypeptide case from sponge.
    Description: Test polypeptide net
    Expectation: Success.
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, enable_graph_kernel=True)
    context.set_context(graph_kernel_flags="--enable_expand_ops=Gather \
                        --enable_cluster_ops=TensorScatterAdd,UnSortedSegmentSum,GatherNd \
                        --enable_recompute_fusion=false --enable_parallel_fusion=true")
    simulation = Simulation()
    for steps in range(1000):
        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
            nb14_lj_energy_sum, nb14_cf_energy_sum, lj_energy_sum, ee_ene, _ = simulation()
        if steps == 1000:
            print(temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, \
                                  sigma_of_dihedral_ene, nb14_lj_energy_sum, nb14_cf_energy_sum, lj_energy_sum, ee_ene)
    assert 250 < temperature < 350
    assert -7500 < total_potential_energy < -6500
    assert 800 < sigma_of_bond_ene < 1300
    assert 10 < sigma_of_dihedral_ene < 25
    assert 3 < nb14_lj_energy_sum < 9
    assert 130 < nb14_cf_energy_sum < 220
    assert 1200 < lj_energy_sum < 1800
    assert -12000 < ee_ene < -7000
