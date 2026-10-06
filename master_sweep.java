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
            BufferedReader configReader = new BufferedReader(new FileReader("sweep_config.txt"));
            String[] targetReports = configReader.readLine().split(",");
            String[] targetFieldFunctions = configReader.readLine().split(",");
            String outputDir = configReader.readLine(); 
            configReader.close();
            
            BufferedReader matrixReader = new BufferedReader(new FileReader("sweep_matrix.csv"));
            String headerLine = matrixReader.readLine();
            String[] parametersToSweep = headerLine.split(",");
            String line = matrixReader.readLine(); 
            matrixReader.close();

            if (line == null) return;
            
            sim.println("==================================================");
            sim.println("🚀 STARTING AUTOMATED CFD RUN");
            sim.println("==================================================");

            // ==========================================
            // PHASE 1: AUTO-TAGGER (LEFT SIDE)
            // ==========================================
            sim.println("🔍 Scanning imported geometry names...");
            for (GeometryPart part : sim.get(GeometryPartManager.class).getObjects()) {
                String name = part.getPresentationName().toLowerCase();
                
                if (name.contains("mirror") || name.contains("copy")) continue;
                
                sim.println("   -> Found Part: '" + part.getPresentationName() + "'");

                if (name.contains("front wheel")) {
                    part.setPresentationName("Front Left Wheel");
                    applyTag(sim, part, "Front Wheel");
                } else if (name.contains("rear wheel")) {
                    part.setPresentationName("Rear Left Wheel");
                    applyTag(sim, part, "Rear Wheel");
                } else if (name.contains("chassis")) {
                    part.setPresentationName("Chassis");
                    applyTag(sim, part, "Chassis");
                } else if (name.contains("front wing")) {
                    part.setPresentationName("Front Wing");
                    applyTag(sim, part, "Front Wing");
                } else if (name.contains("rear wing")) {
                    part.setPresentationName("Rear Wing");
                    applyTag(sim, part, "Rear Wing");
                } else if (name.contains("floor")) {
                    part.setPresentationName("Floor");
                    applyTag(sim, part, "Floor");
                }
            }
            
            sim.println("⚙️ Executing Part Operations (Mirrors, Transforms)...");
            for (MeshOperation op : sim.get(MeshOperationManager.class).getObjects()) {
                String opName = op.getClass().getSimpleName();
                if (!opName.contains("AutoMesh") && !opName.contains("AutomatedMesh")) {
                    try { op.execute(); } catch (Exception e) {}
                }
            }

            // ==========================================
            // PHASE 2: AUTO-TAGGER (RIGHT SIDE)
            // ==========================================
            sim.println("🔍 Scanning for mirrored geometry...");
            for (GeometryPart part : sim.get(GeometryPartManager.class).getObjects()) {
                String name = part.getPresentationName().toLowerCase();
                
                if (name.contains("front left wheel") && (name.contains("mirror") || name.contains("copy") || name.contains(" 2"))) {
                    part.setPresentationName("Front Right Wheel");
                    applyTag(sim, part, "Front Wheel");
                } else if (name.contains("rear left wheel") && (name.contains("mirror") || name.contains("copy") || name.contains(" 2"))) {
                    part.setPresentationName("Rear Right Wheel");
                    applyTag(sim, part, "Rear Wheel");
                }
            }
            
            // 3. APPLY DOE PARAMETERS
            String[] values = line.split(",");
            for (int i = 0; i < parametersToSweep.length; i++) {
                try {
                    ScalarGlobalParameter param = (ScalarGlobalParameter) sim.get(GlobalParameterManager.class).getObject(parametersToSweep[i]);
                    param.getQuantity().setDefinition(values[i]);
                    sim.println("Set " + parametersToSweep[i] + " to " + values[i]);
                } catch (Exception e) {
                    sim.println("Skipped parameter " + parametersToSweep[i] + " (likely an NX parameter).");
                }
            }
            
            sim.println("⚙️ Clearing old solution and meshing...");
            sim.getSolution().clearSolution(Solution.Clear.History, Solution.Clear.Fields, Solution.Clear.Mesh);
            sim.get(MeshOperationManager.class).executeAll();
            
            sim.println("🚀 Running RANS Solver...");
            sim.getSimulationIterator().run();
            
            FileWriter resultsWriter = new FileWriter(outputDir + "/Aero_Map_Results.csv");
            resultsWriter.write(headerLine + "," + String.join(",", targetReports) + "\n");
            
            StringBuilder resultRow = new StringBuilder(line);
            for (String reportName : targetReports) {
                Report report = (Report) sim.getReportManager().getObject(reportName);
                double val = report.getReportMonitorValue();
                resultRow.append(",").append(val);
            }
            resultsWriter.write(resultRow.toString() + "\n");
            resultsWriter.flush();
            resultsWriter.close();
            
            sim.println("📷 Saving Images (Scenes, Layouts, and Residuals)...");
            for (ClientServerObject obj : sim.getSceneManager().getObjects()) {
                if (obj instanceof Scene && obj instanceof Taggable) {
                    Scene scene = (Scene) obj;
                    if (hasCaptureTag(scene)) {
                        scene.printAndWait(resolvePath(outputDir + "/" + scene.getPresentationName() + ".png"), 1, 1920, 1080, true, false);
                    }
                }
            }
            
            try {
                for (ClientServerObject obj : sim.get(LayoutViewManager.class).getObjects()) {
                    if (obj instanceof LayoutView && obj instanceof Taggable) {
                        LayoutView layout = (LayoutView) obj;
                        if (hasCaptureTag(layout)) {
                            layout.printToFile(resolvePath(outputDir + "/" + layout.getPresentationName() + ".png"), 1, 1920, 1080, true, false);
                        }
                    }
                }
            } catch (Exception e) {}
            
            try {
                ResidualPlot residualPlot = (ResidualPlot) sim.getPlotManager().getPlot("Residuals");
                if (residualPlot != null) {
                    residualPlot.encode(resolvePath(outputDir + "/Residuals.png"), "png", 1920, 1080, true, false);
                }
            } catch (Exception e) {}
            
            sim.println("✅ SINGLE RUN COMPLETE!");
            
        } catch (Exception e) {
            sim.println("🚨 MACRO CRASHED: " + e.getMessage());
        }
    }
    
    private void applyTag(Simulation sim, GeometryPart part, String tagName) {
        TagManager tagMgr = sim.get(TagManager.class);
        if (tagMgr.has(tagName)) {
            Tag tag = (Tag) tagMgr.getObject(tagName);
            part.getTagGroup().add(tag);
            sim.println("   [SUCCESS] Tagged '" + part.getPresentationName() + "' with [" + tagName + "]");
        } else {
            sim.println("   [WARNING] Tag [" + tagName + "] does not exist in the simulation!");
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
