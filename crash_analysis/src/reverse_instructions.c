#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include "global.h"
#include "disassemble.h"
#include "insthandler.h"
#include "reverse_exe.h"
#include "analyze_result.h"
#include "bin_alias.h"
#include "re_runtime.h"
#include "common.h"

unsigned long reverse_instructions(void){

	unsigned index; 
	re_list_t *curinst, *previnst; 
	re_list_t *entry; 

	re_list_t re_deflist, re_uselist, re_instlist;  	

	//init the main linked list
	INIT_LIST_HEAD(&re_ds.head.list);
	INIT_LIST_HEAD(&re_ds.head.umemlist);
	INIT_LIST_HEAD(&re_ds.head.memlist);

	LOG_RUNTIME("Finish Loading Data");

	re_ds.resolving = false;
	re_ds.alias_module = NO_ALIAS_MODULE;

	for(index = 0; index < re_ds.instnum; index++){

		if (verify_useless_inst(re_ds.instlist + index)) {
			continue;
		}

		//insert the instruction data into the current linked list
		curinst = add_new_inst(index);
		if( !curinst){
			assert(0);
		}

		print_instnode(curinst->node);

		LOG(stdout, "\n------------------Start of one instruction analysis-----------------\n");

		int handler_index = insttype_to_index(re_ds.instlist[index].type);

		if (handler_index >= 0) {
			inst_handler[handler_index](curinst);
		} else {
			LOG(stdout, "instruction type %x\n", re_ds.instlist[index].type);
			assert(0);
		}

		print_info_of_current_inst(curinst);
		LOG(stdout, "------------------ End of one instruction analysis------------------\n");
	}

	LOG_RUNTIME("Finish The First Round of Reverse Execution");

	//re_statistics();

	//LOG_RUNTIME("Finish Reverse Execution Evaluation");

	// directly calculate truth_address of every memory
	// this must be executed before evaluation method
	calculate_truth_address();

	//print_corelist(&re_ds.head);

	// find all the read/write accesses related to one specific address in the trace
	// find_access_related_address(0xb525d168);

#if defined (ALIAS_MODULE) && (ALIAS_MODULE == NO_MODULE)
	re_statistics();

	LOG_RUNTIME("Finish Reverse Execution Evaluation");

	//analyze_corelist();

	destroy_corelist();

	return 0;
#endif

#if defined (ALIAS_MODULE) && (ALIAS_MODULE == HT_MODULE)
	re_ds.alias_module = HYP_TEST_MODULE;
#endif

	load_region_from_DL(get_region_path());
	LOG_RUNTIME("Finish Loading Region from DeepLearning Data");

#if defined (ALIAS_MODULE) && (ALIAS_MODULE == GT_MODULE)
	re_ds.alias_module = GR_TRUTH_MODULE;
#endif

	//make another assumption here
	//We assume the registers are recorded at the begining of the trace
	//this somehow makes sense

	list_for_each_entry(entry, &re_ds.head.list, list) {

		INIT_LIST_HEAD(&re_deflist.deflist);
		INIT_LIST_HEAD(&re_uselist.uselist);
		INIT_LIST_HEAD(&re_instlist.instlist);	
		
		if(entry->node_type != UseNode)
			continue; 

		init_reg_use(entry, &re_uselist);	
		re_resolve(&re_deflist, &re_uselist, &re_instlist);
	}

	one_round_of_re();

	LOG_RUNTIME("Finish The Second Round of Reverse Execution");

	//re_statistics();
	//LOG_RUNTIME("Finish Reverse Execution Evaluation");

	re_statistics();

	LOG_RUNTIME("Finish Reverse Execution Evaluation");

	//print_corelist(&re_ds.head);

#if defined (ALIAS_MODULE) && (ALIAS_MODULE == HT_MODULE) && defined(DL_AST)
	ratio_noalias_pair_ht();
	LOG_RUNTIME("Finish No Alias Pair Evaluation");
#endif

	analyze_corelist();
	print_umemlist(&re_ds.head);
	LOG_RUNTIME("Finish Backward Taint Analysis");

	destroy_corelist();

	return 0; 
}
