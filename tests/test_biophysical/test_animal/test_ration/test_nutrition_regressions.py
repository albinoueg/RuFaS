"""Nutrition regression tests using synthetic data and the standard unittest runner."""
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch, Mock
from RUFAS.biophysical.animal.herd_manager import HerdManager
from RUFAS.biophysical.animal.ration.ration_manager import RationManager
from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination

import numpy as np
from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements, NutritionSupply, NutritionEvaluationResults
from RUFAS.biophysical.animal.ration.calf_ration_manager import CalfRationManager
from scipy.optimize import OptimizeResult
from RUFAS.biophysical.animal.ration.amino_acid import EssentialAminoAcidRequirements
from RUFAS.biophysical.animal.nutrients.nutrition_evaluator import NutritionEvaluator
from RUFAS.biophysical.animal.nutrients.nutrition_supply_calculator import NutritionSupplyCalculator
from RUFAS.biophysical.animal.ration.ration_optimizer import RationOptimizer
from RUFAS.data_structures.feed_storage_to_animal_connection import NutrientStandard


class NutritionRegressionTests(unittest.TestCase):
    def setUp(self):
        amino = EssentialAminoAcidRequirements(**{f.name: 0.0 for f in fields(EssentialAminoAcidRequirements)})
        self.requirements = NutritionRequirements(
            maintenance_energy=10, growth_energy=2, pregnancy_energy=1,
            lactation_energy=5, activity_energy=0, metabolizable_protein=100,
            calcium=10, phosphorus=5, process_based_phosphorus=0,
            dry_matter=10, essential_amino_acids=amino,
        )
        kwargs = {f.name: 0.0 for f in fields(NutritionSupply) if f.init}
        kwargs.update(lactation_energy=18, metabolizable_protein=110,
                      calcium=10, phosphorus=5, dry_matter=10,
                      ndf_supply=3, forage_ndf_supply=2, fat_supply=0.3)
        self.supply = NutritionSupply(**kwargs)

    def evaluate(self, is_cow=True):
        return NutritionEvaluator.evaluate_nutrition_supply(
            self.requirements, self.supply, is_cow, NutrientStandard.NASEM)

    def test_nasem_common_energy_validates_cow_and_heifer(self):
        self.assertTrue(self.evaluate()[0])
        self.assertTrue(self.evaluate(False)[0])

    def test_nasem_total_deficit_cannot_be_hidden_by_component_surpluses(self):
        self.supply.lactation_energy = 17.99
        valid, evaluation = self.evaluate(False)
        self.assertFalse(valid)
        self.assertAlmostEqual(evaluation.total_energy, -0.01)

    def test_fat_below_ceiling_is_valid_above_ceiling_is_invalid(self):
        self.assertEqual(self.evaluate()[1].fat_percent, 0)
        self.supply.fat_supply = 0.8
        self.assertFalse(self.evaluate()[0])
        self.assertLess(self.evaluate()[1].fat_percent, 0)

    def test_ndf_oversupply_is_reported_negative(self):
        self.supply.ndf_supply = 5
        self.assertFalse(self.evaluate()[0])
        self.assertLess(self.evaluate()[1].ndf_percent, 0)

    def test_rounding_does_not_hide_material_phosphorus_deficit(self):
        self.supply.phosphorus = 5 - 1e-5
        self.assertTrue(self.evaluate()[0])
        self.supply.phosphorus = 4.99
        self.assertFalse(self.evaluate()[0])

    def test_lactation_flag_checks_its_own_margin(self):
        e = self.evaluate()[1]
        e.total_energy = -1
        e.lactation_energy = 1
        self.assertTrue(e.report['lactation_energy_acceptable'])
        self.assertFalse(e.report['total_energy_acceptable'])

    def test_nrc_retains_separate_energy_supply_fields(self):
        self.supply.maintenance_energy = 20
        self.supply.growth_energy = 6
        self.supply.lactation_energy = 20
        valid, _ = NutritionEvaluator.evaluate_nutrition_supply(self.requirements, self.supply, True)
        self.assertTrue(valid)
        self.supply.maintenance_energy = 0
        self.assertFalse(NutritionEvaluator.evaluate_nutrition_supply(self.requirements, self.supply, True)[0])

    def test_energy_constraint_uses_mean_of_individual_nonlinear_supply(self):
        config = SimpleNamespace(
            animal_nasem_states=[(1, 0, 0), (3, 0, 0)],
            animal_requirements=SimpleNamespace(total_energy_requirement=0),
            pen_average_body_weight=2, pen_average_enteric_methane=0, pen_average_urine_nitrogen=0,
        )
        with patch.object(RationOptimizer, 'convert_decision_vec_to_feeds', return_value=[]), \
             patch.object(NutritionSupplyCalculator, 'calculate_NASEM_metabolizable_energy', side_effect=lambda **kw: kw['body_weight']**2), \
             patch.object(NutritionSupplyCalculator, 'calculate_NASEM_net_energy', side_effect=lambda **kw: kw['total_metabolizable_energy']):
            self.assertEqual(RationOptimizer.NASEM_net_energy_constraint(np.array([1.0]), config), 5)
            config.animal_nasem_states = None
            self.assertEqual(RationOptimizer.NASEM_net_energy_constraint(np.array([1.0]), config), 4)

    def test_protein_constraint_uses_individual_weights_and_preserves_requirement(self):
        config = SimpleNamespace(
            animal_nasem_states=[(1, 0, 0), (3, 0, 0)],
            animal_requirements=SimpleNamespace(metabolizable_protein=7),
            initial_protein_requirement=8, pen_average_body_weight=2,
        )
        with patch.object(RationOptimizer, 'convert_decision_vec_to_feeds', return_value=[]), \
             patch.object(NutritionSupplyCalculator, 'calculate_nutrient_intake_discount', return_value=1), \
             patch.object(NutritionSupplyCalculator, 'calculate_metabolizable_protein_supply', side_effect=lambda **kw: kw['body_weight']**2):
            self.assertEqual(RationOptimizer._calculate_protein_constraint_parameters(np.array([1.0]), config, False), (5, 7))
            self.assertEqual(RationOptimizer._calculate_protein_constraint_parameters(np.array([1.0]), config, True), (5, 8))
            config.animal_nasem_states = None
            self.assertEqual(RationOptimizer._calculate_protein_constraint_parameters(np.array([1.0]), config, False), (4, 7))

    def test_applied_ration_preserves_solver_precision(self):
        result = OptimizeResult(success=True, x=np.array([1.123456789123]))
        ration = RationOptimizer.make_ration_from_solution([SimpleNamespace(rufas_id=51)], result)
        self.assertEqual(ration[51], 1.123456789123)

    def test_digestible_calf_protein_is_a_mass_and_scales_with_intake(self):
        calc = CalfRationManager.calculate_apparent_digestible_protein_intake
        self.assertEqual(calc(0, 0), 0)
        self.assertAlmostEqual(calc(0.1, 0.05), 130.5)
        self.assertAlmostEqual(calc(0.2, 0.1), 261)

    def test_unavailable_calf_evaluation_is_not_reported_as_accepted_or_rejected(self):
        evaluation = NutritionEvaluationResults.make_empty_evaluation_results()
        evaluation.evaluation_available = False
        self.assertFalse(evaluation.is_valid_heifer_ration)
        self.assertFalse(evaluation.is_valid_cow_ration)
        self.assertFalse(evaluation.report['evaluation_available'])
        self.assertIsNone(evaluation.report['is_valid_heifer_ration'])
        self.assertIsNone(evaluation.report['fat_percent_acceptable'])
        self.assertFalse((evaluation / 2).evaluation_available)
        self.assertFalse((evaluation + self.evaluate()[1]).evaluation_available)

    def test_repopulating_empty_pen_selects_feed_ids_for_its_category(self):
        pen = SimpleNamespace(id=0, animal_combination=AnimalCombination.CALF,
                              current_stocking_density=0, is_populated=False,
                              insert_single_animal_into_animals_in_pen_map=Mock(),
                              set_animal_nutritional_requirements=Mock())
        manager = SimpleNamespace(
            ANIMAL_GROUPING_SCENARIO=SimpleNamespace(find_animal_combination=lambda a: AnimalCombination.CALF),
            pens_by_animal_combination={AnimalCombination.CALF:[pen]},
            is_ration_defined_by_user=False, animal_to_pen_id_map={},
            _find_pen_available_feeds=lambda feeds, ids: [f for f in feeds if f.rufas_id in ids],
            _reformulate_ration_single_pen=Mock(),
        )
        feeds = [SimpleNamespace(rufas_id=202), SimpleNamespace(rufas_id=216), SimpleNamespace(rufas_id=44)]
        with patch.object(RationManager, 'get_ration_feeds', return_value=[202,216]) as selected:
            HerdManager._add_animal_to_pen_and_id_map(
                manager, SimpleNamespace(id=101), feeds, SimpleNamespace(mean_air_temperature=25), 42)
        selected.assert_called_once_with(AnimalCombination.CALF)
        selected_feeds = manager._reformulate_ration_single_pen.call_args.kwargs['pen_available_feeds']
        self.assertEqual([f.rufas_id for f in selected_feeds], [202,216])
        self.assertEqual(manager.animal_to_pen_id_map[101], 0)

    def test_empty_calf_feed_configuration_has_explicit_error(self):
        pen = SimpleNamespace(id=0, animal_combination=AnimalCombination.CALF)
        with self.assertRaisesRegex(ValueError, 'No configured calf feeds'):
            HerdManager._reformulate_ration_single_pen(SimpleNamespace(), pen, [], 25, 42)


if __name__ == '__main__':
    unittest.main()
