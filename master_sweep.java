package macro;

import java.util.*;
import java.io.*;
import star.common.*;
import star.base.report.*;
import star.base.neo.*;
import star.meshing.*;
import star.vis.*; 

public class master_sweep extends StarMacro {

    public void execute() {
        Simulation sim = getActiveSimulation();
        
        try {
            // 1. Read Config File
            BufferedReader configReader = new BufferedReader(new FileReader("sweep_config.txt"));
            String[] targetReports = configReader.readLine().split(",");
            String[] targetFieldFunctions = configReader.readLine().split(",");
            String outputDir = configReader.readLine(); 
            configReader.close();
            
            // 2. Prepare Output Results CSV
            FileWriter resultsWriter = new FileWriter(outputDir + "/Aero_Map_Results.csv");
            
            // 3. Read the Run Matrix
            BufferedReader matrixReader = new BufferedReader(new FileReader("sweep_matrix.csv"));
            String headerLine = matrixReader.readLine();
            String[] parametersToSweep = headerLine.split(",");
            
            // Write Header
            resultsWriter.write("Run_ID," + headerLine + "," + String.join(",", targetReports) + "\n");
            
            // 4. THE MAIN SWEEP LOOP
            String line;
            int runId = 1;
            
            while ((line = matrixReader.readLine()) != null) {
                sim.println("==================================================");
                sim.println("🚀 STARTING SWEEP RUN " + runId);
                sim.println("==================================================");
                
                String[] values = line.split(",");
                
                // --- CREATE DYNAMIC SUBFOLDER ---
                StringBuilder folderName = new StringBuilder(outputDir + "/Run_" + runId);
                for (int i = 0; i < parametersToSweep.length; i++) {
                    folderName.append("_").append(parametersToSweep[i]).append("_").append(values[i]);
                }
                String runFolder = folderName.toString();
                new File(runFolder).mkdirs(); 
                
                // A. Apply Parameters
                for (int i = 0; i < parametersToSweep.length; i++) {
                    ScalarGlobalParameter param = (ScalarGlobalParameter) sim.get(GlobalParameterManager.class).getObject(parametersToSweep[i]);
                    param.getQuantity().setDefinition(values[i]);
                    sim.println("Set " + parametersToSweep[i] + " to " + values[i] + " (Base SI)");
                }
                
                // B. Clear Old Physics and Remesh
                sim.println("Clearing old solution, fields, and mesh...");
                sim.getSolution().clearSolution(Solution.Clear.History, Solution.Clear.Fields, Solution.Clear.Mesh);
                sim.get(MeshOperationManager.class).executeAll();
                
                // C. Solve Physics
                sim.println("Running RANS Solver...");
                sim.getSimulationIterator().run();
                
                // D. Extract Scalar Reports
                StringBuilder resultRow = new StringBuilder();
                resultRow.append(runId).append(",").append(line);
                
                for (String reportName : targetReports) {
                    Report report = (Report) sim.getReportManager().getObject(reportName);
                    double val = report.getReportMonitorValue();
                    resultRow.append(",").append(val);
                }
                
                resultsWriter.write(resultRow.toString() + "\n");
                resultsWriter.flush(); 
                
                // E. Save Hardcopy Images to the SUBFOLDER
                sim.println("Saving Images (Scenes, Layouts, and Residuals)...");
                
                // E1. Tagged Scenes
                for (ClientServerObject obj : sim.getSceneManager().getObjects()) {
                    if (obj instanceof Scene && obj instanceof Taggable) {
                        Scene scene = (Scene) obj;
                        if (hasCaptureTag(scene)) {
                            scene.printAndWait(resolvePath(runFolder + "/" + scene.getPresentationName() + ".png"), 1, 1920, 1080, true, false);
                            sim.println("Saved Scene: " + scene.getPresentationName());
                        }
                    }
                }
                
                // E2. Tagged Layouts
                try {
                    for (ClientServerObject obj : sim.get(LayoutViewManager.class).getObjects()) {
                        if (obj instanceof LayoutView && obj instanceof Taggable) {
                            LayoutView layout = (LayoutView) obj;
                            if (hasCaptureTag(layout)) {
                                layout.printToFile(resolvePath(runFolder + "/" + layout.getPresentationName() + ".png"), 1, 1920, 1080, true, false);
                                sim.println("Saved Layout: " + layout.getPresentationName());
                            }
                        }
                    }
                } catch (Exception e) {
                    sim.println("Note: Layout export failed - " + e.getMessage());
                }
                
                // E3. Residuals Plot (Automatic capture, no tag required)
                try {
                    ResidualPlot residualPlot = (ResidualPlot) sim.getPlotManager().getPlot("Residuals");
                    if (residualPlot != null) {
                        residualPlot.encode(resolvePath(runFolder + "/Residuals.png"), "png", 1920, 1080, true, false);
                        sim.println("Saved Plot: Residuals");
                    }
                } catch (Exception e) {
                    sim.println("Note: Residuals plot export failed.");
                }
                
                runId++;
            }
            
            matrixReader.close();
            resultsWriter.close();
            sim.println("✅ BATCH SWEEP COMPLETE!");
            
        } catch (Exception e) {
            sim.println("🚨 MACRO CRASHED: " + e.getMessage());
        }
    }
    
    @SuppressWarnings("deprecation")
    private boolean hasCaptureTag(Taggable obj) {
        TagGroup tagGroup = obj.getTagGroup();
        if (tagGroup != null) {
            for (Tag tag : tagGroup.getObjects()) {
                if (tag.getPresentationName().toLowerCase().contains("capture")) {
                    return true;
                }
            }
        }
        return false;
    }
}