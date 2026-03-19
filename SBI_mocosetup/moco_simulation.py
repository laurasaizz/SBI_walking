import opensim as osm
from sample_human import sample_human
from sample_human import adjust_pelvis_ty_by_height
def run_moco_simulation():
    # --------------------------------------------------
    # Pfade
    # --------------------------------------------------
    reference_file = "normal.mot"   
    solution_file = "subject02_torque_tracking_solution.sto"

    # --------------------------------------------------
    # 2) Modell laden
    # --------------------------------------------------
    print("Lade Modell ...")
    model, sampled_height = sample_human("test_human")
    reference_file = adjust_pelvis_ty_by_height(reference_file, sampled_height, output_mot= "scaled_motion.mot")

    # Alle Koordinaten entsperren
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        coord.setDefaultLocked(False)

    model.finalizeConnections()

    # --------------------------------------------------
    # 3) ModelProcessor (TORQUE-DRIVEN SETUP)
    # --------------------------------------------------
    print("Erzeuge ModelProcessor für Torque-Drive ...")
    modelProcessor = osm.ModelProcessor(model)

    # SCHRITT A: Alle Muskeln entfernen
    modelProcessor.append(osm.ModOpRemoveMuscles())

    # SCHRITT B: Drehmoment-Aktuatoren (Reserves) hinzufügen
    # Wir fügen starke Reserven (Torque Actuators) für alle Freiheitsgrade hinzu.
    # 250 Nm ist ein guter Startwert für die meisten Gelenke.
    modelProcessor.append(osm.ModOpAddReserves(250))

    # --------------------------------------------------
    # 4) MocoTrack Setup
    # --------------------------------------------------
    print("Setze MocoTrack auf ...")
    track = osm.MocoTrack()
    track.setName("subject02_torque_tracking")
    track.setModel(modelProcessor)

    states_ref = osm.TableProcessor(reference_file)
    # Falls die Referenz nur Winkel hat, berechnet Moco die Winkelgeschwindigkeiten
    states_ref.append(osm.TabOpLowPassFilter(6)) 
    track.setStatesReference(states_ref)

    # Einstellungen für die Simulation
    track.set_states_global_tracking_weight(10)
    track.set_allow_unused_references(True)
    track.set_track_reference_position_derivatives(True)
    track.set_apply_tracked_states_to_guess(True)

    # Zeitbereich
    track.set_initial_time(0)
    track.set_final_time(1)
    track.set_mesh_interval(0.01) # Etwas feineres Mesh für Stabilität
    track.set_clip_time_range(True)

    # Effort-Gewichtung (Minimierung der Drehmomente)
    track.set_control_effort_weight(0.1)

    # --------------------------------------------------
    # 5) Individuelle State-Gewichte
    # --------------------------------------------------
    print("Setze individuelle Tracking-Gewichte ...")
    weights = osm.MocoWeightSet()

    # Wichtige Gelenke stärker tracken
    joints = [
        "pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list", "pelvis_rotation",
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
        "knee_angle_r", "ankle_angle_r", "knee_angle_l", "ankle_angle_l",
        "lumbar_extension", "lumbar_bending", "lumbar_rotation"
    ]

    for joint in joints:
        weights.cloneAndAppend(osm.MocoWeight(joint, 10.0))

    # Schwache Gelenke (verhindert Instabilität bei fehlenden Daten)
    for side in ["_r", "_l"]:
        weights.cloneAndAppend(osm.MocoWeight(f"subtalar_angle{side}", 0.1))
        weights.cloneAndAppend(osm.MocoWeight(f"mtp_angle{side}", 0.1))

    track.set_states_weight_set(weights)


    # --------------------------------------------------
    # 6) Study & Solver Initialisierung
    # --------------------------------------------------
    print("Initialisiere Study ...")
    study :osm.MocoStudy= track.initialize()
    solver :osm.MocoCasADiSolver= study.initCasADiSolver()
    solver.set_parallel(1) # 0 for no parallel, 1 for all cores,
    solver.set_optim_solver("ipopt")
    solver.set_verbosity(2)
    solver.set_optim_convergence_tolerance(1e-3)
    solver.set_optim_constraint_tolerance(1e-3)

    # Erhöht auf 100, damit die Simulation eine Chance hat zu konvergieren
    # (15 war zu wenig)
    solver.set_optim_max_iterations(100) 

    # --------------------------------------------------
    # 8) Lösen
    # --------------------------------------------------
    print("Starte Optimierung ...")
    solution = study.solve()

    if solution.success():
        print("Speichere Lösung im Speicher...")
    else:
        print("Simulation nicht konvergiert, bereite unsealed Lösung zur Rückgabe vor...")
        # Unseal erlaubt den Zugriff auf Zwischenergebnisse bei Fehlern
        solution.unseal()

    print("Fertig. Lösung wird übergeben.")
    return solution, model