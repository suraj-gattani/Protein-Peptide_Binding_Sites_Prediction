import java.io.*;
import java.util.*;

public class GeneticAlgorithm_feature_selection {

	/*
	 * Parameters for GA
	 */
	private static int chrom_len = 247;
	private static int pop_size = 200;
	static int max_generations = 2;
	static float target_fitness = 10000f;
	static float ElitRate = 0.05f;
	static float CrossOverRate = 0.9f;
	static float MutationRate = 0.5f;
	static double[][] feature_arrays= new double[2110][247];
	
	public class Genome {

		float fitness = 0;
		int[] chromosome = new int[chrom_len];

	}

	public static void main(String[] args) throws IOException {

		try {
			File currentDir = new File(new File(".").getAbsolutePath());
			String curr_dir_path = currentDir.getCanonicalPath();
			String output = curr_dir_path + "/result.txt";
			File op_file = new File(output);
			BufferedWriter result_writer = BufferReaderAndWriter.getWriter(op_file);
			Genome[] genoA = new Genome[pop_size];
			Genome[] genoB = new Genome[pop_size];
			
			feature_array_creation(curr_dir_path);
			
			GeneticAlgorithm_feature_selection optEgy = new GeneticAlgorithm_feature_selection();
			GeneticAlgorithm_feature_selection.Genome genoTemp1 = optEgy.new Genome();
			GeneticAlgorithm_feature_selection.Genome genoTemp2 = optEgy.new Genome();

			int generations = 0;
			int new_popn_chromosome_index = 0;
			float obtained_fitness = 0;
			System.out.println("GA started");
			result_writer.write("GA started");
			result_writer.newLine();

			optEgy.initializePop(genoA, result_writer, curr_dir_path);
			sortPopWithFitness(genoA, genoTemp1);

			System.out.println("fitness after sorting");
			for (int j = 0; j < pop_size; j++) {

				System.out.println(genoA[j].fitness);
				for (int t = 0; t < chrom_len; t++) {

					System.out.print(genoA[j].chromosome[t]);

				}

				System.out.println();

			}

			System.out
					.println("Finished initializing population and sorting population according to fitness");

			while ((generations < max_generations)
					&& (obtained_fitness < target_fitness)) {

				generations += 1;
				new_popn_chromosome_index = doElit(genoA, genoB);
				int elit_index = new_popn_chromosome_index + 1; // index -- one
																// greater than
																// elit index --
																// because
																// letter we
																// don't have to
																// calculate
																// fitness for
																// chromosomes
																// before this
																// index as they
																// are already
																// calculate
																// while elit
				System.out.println("after elit index "
						+ new_popn_chromosome_index); // this index will be 1
														// less than number of
														// elit as return value
														// is num_elit-1

				// ===================== Start of Crossover
				// =======================//

				float totalFitness = 0;
				float rand1;
				float rand2;
				int firstSelectionIndex;
				int secondSelectionIndex;
				int cross_over_point;
				int num_of_Crossover = (int) ((CrossOverRate * pop_size) / 2);

				for (int i = 0; i < pop_size; i++) {

					totalFitness += genoA[i].fitness;

				}

				System.out.println("totalFitness " + totalFitness);

				for (int I = 0; I < num_of_Crossover; I++) {
					Genome tempCand1 = null, tempCand2 = null;
					Genome tempNewPopCand1 = optEgy.new Genome(), tempNewPopCand2 = optEgy.new Genome();

					rand1 = randomBetweenTwoNumber(0.0f, totalFitness);
					System.out.println("first random no: " + rand1);

					firstSelectionIndex = 0;
					while (rand1 > 0) {

						rand1 = rand1 - genoA[firstSelectionIndex].fitness;
						firstSelectionIndex++;

					}
					tempCand1 = genoA[firstSelectionIndex - 1];
					System.out.println("First Selection Index "
							+ (firstSelectionIndex - 1));
					
					rand2 = randomBetweenTwoNumber(0.0f, totalFitness);
					System.out.println("second random no: " + rand2);

					secondSelectionIndex = 0;
					while (rand2 > 0) {

						rand2 = rand2 - genoA[secondSelectionIndex].fitness;
						secondSelectionIndex++;
					}
					tempCand2 = genoA[secondSelectionIndex - 1];
					System.out.println("Second Selection Index "
							+ (secondSelectionIndex - 1));
					
					cross_over_point = (int) Math.ceil(chrom_len
							* randomBetweenTwoNumber(0.0f, 1.0f));

					
					for (int l = 0; l < cross_over_point; l++) {

						tempNewPopCand1.chromosome[l] = tempCand1.chromosome[l];
						tempNewPopCand2.chromosome[l] = tempCand2.chromosome[l];

					}

					for (int m = cross_over_point; m < chrom_len; m++) {

						tempNewPopCand1.chromosome[m] = tempCand2.chromosome[m];
						tempNewPopCand2.chromosome[m] = tempCand1.chromosome[m];

					}
					

					new_popn_chromosome_index++; // first increase the index
													// because elit returns the
													// value which is (number of
													// elit-1)
					genoB[new_popn_chromosome_index] = tempNewPopCand1;
					new_popn_chromosome_index++;
					genoB[new_popn_chromosome_index] = tempNewPopCand2;

				}

				System.out.println("after finishing cross-over index: "
						+ new_popn_chromosome_index);

				// ===================== End of Crossover
				// =======================//

				// ======================== Start Fill The Rest
				// =================//
				int fill_rest_count = 0;
				for (int p = new_popn_chromosome_index + 1; p < pop_size; p++) {
					genoB[p] = new GeneticAlgorithm_feature_selection().new Genome();
					generateChromosome(genoB[p], p, 2, result_writer,
							curr_dir_path); // popType 2 means generate only
											// chromosome without fitness
					fill_rest_count++;

				}

				System.out.println("after filling in rest "
						+ (new_popn_chromosome_index + fill_rest_count));

				// exit(0);

				// ======================== End Fill The Rest ==================
				// //

				// ======================== Start of Mutation ==================
				// //

				int num_of_mutation = (int) (MutationRate * pop_size);
				int num_of_Elit = (int) (ElitRate * pop_size);
				int chromosome_index_to_mutate = 0;

				for (int q = 0; q < num_of_mutation; q++) {

					chromosome_index_to_mutate = (int) (Math
							.ceil((pop_size - num_of_Elit)
									* randomBetweenTwoNumber(0.0f, 1.0f)) + num_of_Elit);
					
					int mutation_index = (int) Math.ceil(chrom_len
							* randomBetweenTwoNumber(0.0f, 1.0f));
					genoTemp2 = genoB[chromosome_index_to_mutate - 1];
					if (genoTemp2.chromosome[mutation_index - 1] == 0) {

						genoTemp2.chromosome[mutation_index - 1] = 1;

					} else {

						genoTemp2.chromosome[mutation_index - 1] = 0;

					}

					genoB[chromosome_index_to_mutate - 1] = genoTemp2;

				}

				// ======================== End of Mutation ==================
				// //

				Genome[] temp = genoA; // swaping array of Genome type object
				genoA = genoB;
				genoB = temp;

				System.out.println("Started calculating fitness inside GA");

				for (int r = elit_index; r < pop_size; r++) { // calculate
																// fitness of
																// new
																// population
					System.out.println(Arrays.toString(genoA[r].chromosome));
					genoA[r].fitness = calculateFitness(genoA[r].chromosome,
							result_writer, curr_dir_path);
					System.out.println(genoA[r].fitness);
				
				}

				sortPopWithFitness(genoA, genoTemp1); // sort population
														// according to fitness
				System.out.println("fitness after sorting");
				for (int j = 0; j < pop_size; j++) {

					System.out.println(genoA[j].fitness);
					for (int t = 0; t < chrom_len; t++) {

						System.out.print(genoA[j].chromosome[t]);

					}

					System.out.println();

				}
				obtained_fitness = genoA[0].fitness;
				System.out.println("obtained fitness at the end of "
						+ generations + " generation " + "\t"
						+ obtained_fitness);

				// just print result in a file
				result_writer.write("obtained fitness at the end of "
						+ generations + " generation " + "\t"
						+ obtained_fitness);
				result_writer.newLine();
				result_writer.flush();

				for (int s = 0; s < chrom_len; s++) {

					result_writer.write(Integer
							.toString(genoA[0].chromosome[s]));

				}
				result_writer.newLine();
				result_writer.flush();

			}

			System.out.println("obtained fitness at the end of " + generations
					+ " generation " + "\t" + obtained_fitness);

			// print final result in a file
			result_writer.write("obtained fitness at the end of " + generations
					+ " generation " + "\t" + obtained_fitness);
			result_writer.newLine();

			for (int t = 0; t < chrom_len; t++) {

				result_writer.write(Integer.toString(genoA[0].chromosome[t]));

			}

			result_writer.newLine();
			result_writer.flush();
			result_writer.close();
			

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}

	public static void sortPopWithFitness(Genome[] genoA, Genome genoTemp1) {

		for (int c = 0; c < (pop_size - 1); c++) {
			for (int d = c + 1; d < pop_size; d++) {
				if ((genoA[c].fitness) < (genoA[d].fitness)) {
					// printf("%s\n", "I am inside sorting loop");
					genoTemp1 = genoA[d];
					genoA[d] = genoA[c];
					genoA[c] = genoTemp1;

				}
			}
		}

	}

	static int doElit(Genome[] genoA, Genome[] genoB) {

		int num_elit = (int) (ElitRate * pop_size);

		if (num_elit == 0) {

			System.out
					.println("either population size is too small or elit rate is too small");

		}

		for (int i = 0; i < num_elit; i++) {
			genoB[i] = genoA[i];
		}

		return (num_elit - 1); // because index starts from 0
	}

	public static float randomBetweenTwoNumber(float min, float max) {
		Random rand = new Random();
		return min + rand.nextFloat() * (max - min);

	}

	void initializePop(Genome[] genoA, BufferedWriter result_writer,
			String curr_dir_path) throws Exception {

		for (int i = 0; i < pop_size; i++) {

			genoA[i] = new GeneticAlgorithm_feature_selection().new Genome();
			generateChromosome(genoA[i], i, 1, result_writer, curr_dir_path);
			
			

		}

	}

	static void generateChromosome(Genome geno, int chromo_num, int pop_type,
			BufferedWriter result_writer, String curr_dir_path)
			throws Exception {

		if (pop_type == 1) {

			Random ran = new Random();

			for (int j = 0; j < chrom_len; j++) {

				if (ran.nextBoolean() == true) {

					geno.chromosome[j] = 1;

				} else {

					geno.chromosome[j] = 0;

				}
				System.out.print(geno.chromosome[j]);

			}
			System.out.println();
			
			geno.fitness = calculateFitness(geno.chromosome, result_writer,
					curr_dir_path);
			System.out.println("generated fitness " + geno.fitness);

		} else if (pop_type == 2) {

			Random ran = new Random();
			for (int j = 0; j < chrom_len; j++) {

				if (ran.nextBoolean() == true) {

					geno.chromosome[j] = 1;

				} else {

					geno.chromosome[j] = 0;

				}

			}

		}

	}

	
	
	public static float calculateFitness(int[] chromo,
			BufferedWriter result_writer, String curr_dir_path)
			throws Exception {
		String s;
		float acc=0;
		float auc=0;
		float mcc=0;
		float obtainedFitness = 0;
		feature_file_creation(chromo,curr_dir_path);
		Process p = Runtime.getRuntime().exec("python XGBoostCV10.py");
		BufferedReader output_xgb= new BufferedReader(new InputStreamReader(p.getInputStream()));
		while ((s=output_xgb.readLine()) != null)
		{
			if(s.charAt(0)=='R' && s.charAt(4)=='l')
			{
				String[] values=s.split(",");
				auc = Float.parseFloat(values[1]);
				acc = Float.parseFloat(values[2]);
				mcc = Float.parseFloat(values[3]);
			}
		}
		obtainedFitness= auc + acc + mcc;
		result_writer.write("Obtained Fitness: " + obtainedFitness);
		result_writer.newLine();
		result_writer.flush();
		return obtainedFitness;
	}

	public static void feature_file_creation(int[] chromo_current, String curr_dir_path) throws Exception
	{
		String file_name=curr_dir_path+"/feature_file.csv";
	
		String[] temp=new String[2110];
		BufferedWriter writer = new BufferedWriter(new FileWriter(file_name));
		File file = new File(file_name);
		if (file.createNewFile())
		{
			System.out.println("File is created!");
		} else {
			PrintWriter pwriter = new PrintWriter(file);
			pwriter.print("");
			pwriter.close();
			System.out.println("File already exists.");
		}
		for(int j=0;j<chrom_len;j++)
		{
			if(chromo_current[j]==1)
			{
				for(int i=0;i<2110;i++)
				{
					if(j==0)
					{
						temp[i]="";
					}
					temp[i]+=feature_arrays[i][j]+",";
					
				}
			}
			
		}
		for(int j=0;j<2110;j++)
		{
		writer.write(temp[j].substring(0,temp[j].length()-1));
		writer.write("\n");
		}
		writer.close();
	}
	
	public static void feature_array_creation(String path) throws Exception
	{
		File file= new File(path+"/feature_file_test_ws1.csv");
		BufferedReader br=new BufferedReader(new FileReader(file));
		String str;
		int i=0;
		while((str=br.readLine())!=null)
		{
			String[] values=str.split("\\s*,\\s*");
			for(int j=0;j<values.length-1;j++)
			{
				feature_arrays[i][j]=Double.parseDouble(values[j]);
			}
			i++;
		}
		
	}
}
