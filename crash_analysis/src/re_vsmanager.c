#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include "global.h"
#include "disassemble.h"
#include "insthandler.h"
#include "reverse_exe.h"
#include "inst_opd.h"
#include "bin_alias.h"
#include "re_runtime.h"
#include "inst_data.h"
#include "common.h"

static unsigned long truth_address(re_list_t *exp);

static unsigned long get_regvalue_from_pin(re_list_t *reg);

static bool is_effective_reg(re_list_t *entry);

static bool is_effective_exp(re_list_t *entry);

bool dl_region_verify_noalias(re_list_t *exp1, re_list_t *exp2) {
	int reg1 = GET_DL_REGION(exp1);
	int reg2 = GET_DL_REGION(exp2);

	if (reg1 != reg2)
		return true;
	else
		return false;
}

static void set_dl_region(re_list_t *entry, int region) {
// create one field in use_node_t and def_node_t to store dl_region from deeplearning

	region_type type;

	switch (region) {
	case 0:
		type = GLOBAL_REGION;
		break;
	case 1:
		type = HEAP_REGION;
		break;
	case 2:
		type = STACK_REGION;
		break;
	case 3:
		type = OTHER_REGION;
		break;
	default:
		assert("No Such Region Type" && 0);
		break;
	}

        if (entry->node_type == DefNode) {
		CAST2_DEF(entry->node)->dl_region = type;
        } else if (entry->node_type == UseNode) {
		CAST2_USE(entry->node)->dl_region = type;
        } else {
                assert("InstNode could not get here" && 0);
        }
}

void load_region_num(re_list_t *instnode, dlregion_list_t *dlreglist) {
        re_list_t *entry;
        size_t i, num = 0;
	re_list_t *memop[NOPD] = {NULL};

        //x86_insn_t * inst;
        //inst = re_ds.instlist + CAST2_INST(instnode->node)->inst_index;

        list_for_each_entry_reverse(entry, &instnode->list, list) {
                if ((entry->node_type == InstNode) ||
               	    (entry == &re_ds.head)) break;
		// get all the memory operands for current instruction
		if (is_effective_exp(entry)) memop[num++] = entry;
        }

        switch (dlreglist->dlreg_num){
                case 0:
                        break;
                case 1:
                        // only traverse the effective expressions
                        for (i = 0; i < num; i++) {
                                set_dl_region(memop[i], dlreglist->dlreg_list[0]);
                        }
                        break;
                case 2:
                        // only traverse the effective expressions
                        if (num == 2) {
                        	set_dl_region(memop[0], dlreglist->dlreg_list[1]);
                        	set_dl_region(memop[1], dlreglist->dlreg_list[0]);
			}
                        break;
                default:
                        assert(0);
        }
}

// load region in re_ds.dlregionlist and store them into re_valueset field
void load_region_from_DL() {
        re_list_t *entry;
        size_t index = 0;

	//LOG(stderr, "Load region from deep learning\n");
        list_for_each_entry(entry, &re_ds.head.list, list) {
                if (entry->node_type == InstNode){
                        index = CAST2_INST(entry->node)->inst_index;
                        load_region_num(entry, re_ds.dlregionlist + index);
                }
        }
}

#define GET_HT_ADDR(entry) \
	((entry)->node_type == DefNode) ? \
	(CAST2_DEF((entry)->node)->address) : \
	(CAST2_USE((entry)->node)->address) 

bool ht_verify_noalias(re_list_t *exp1, re_list_t *exp2) {
	unsigned addr1 = GET_HT_ADDR(exp1);
	unsigned addr2 = GET_HT_ADDR(exp2);

	if (!addr1 || !addr2)
		return false;
	else
		return (addr1 != addr2);
}


void noalias_pair_num_ht(re_list_t *exp, unsigned long *pt_num, unsigned long *ph_num, unsigned long *err_ph, unsigned long *pd_num, unsigned long *err_pd, unsigned long *hand_num) {
	// initialize the num of noalias pair
	*pt_num = 0;
	*ph_num = 0;
	*pd_num = 0;
	*hand_num = 0;
	*err_ph = 0;
	*err_pd = 0;

	re_list_t *entry;
	x86_op_t *opd;
	bool bool_truth, bool_ht, bool_dl;

	list_for_each_entry(entry, &exp->memlist, memlist) {
		if (entry == &re_ds.head) break;
		if (entry->node_type == UseNode && exp->node_type == UseNode) continue;
		//print_node(exp);
		//print_node(entry);
		bool_truth = truth_verify_noalias(entry, exp);
		bool_ht  = ht_verify_noalias(entry, exp);
		bool_dl  = dl_region_verify_noalias(entry, exp);
		// Truth: Alias, Value Set: No Alias
		if (!bool_truth && bool_ht) {
			print_node(exp);
			print_node(entry);
			(*err_ph)++;
		}
		// Truth: Alias, Value Set: No Alias, DeepLearning: No Alias
		if (!bool_truth && bool_ht && bool_dl) {
			print_node(exp);
			print_node(entry);
			(*err_pd)++;
		}

		if (bool_truth) (*pt_num)++;
		if (bool_dl && bool_ht) (*hand_num)++;
		if (bool_dl || bool_ht) (*pd_num)++;
		if (bool_ht)   (*ph_num)++;
	}
}

void ratio_noalias_pair_ht(){
	unsigned long long ht_noalias = 0, htdl_noalias = 0, truth_noalias = 0, hand_noalias = 0;
	unsigned long long ht_err_noalias = 0, htdl_err_noalias = 0;
	unsigned long htdl_err = 0, ht_err = 0;
	unsigned long tmp_ht = 0, tmp_truth = 0, tmp_htdl = 0, tmp_hand;
	//unsigned long memopd_num = 0;

	re_list_t *entry;
	x86_op_t *opd;
	
	list_for_each_entry(entry, &re_ds.head.memlist, memlist) {
		// print_node(entry);
		// initialize the value inside the function
		noalias_pair_num_ht(entry, &tmp_truth, &tmp_ht, &ht_err, &tmp_htdl, &htdl_err, &tmp_hand);
		//LOG(stdout, "temp valset_noalias %ld ", tmp_vs);
		//LOG(stdout, "temp truth_noalias %ld\n", tmp_truth);
		ht_noalias += tmp_ht;
		hand_noalias += tmp_hand;
		htdl_noalias += tmp_htdl;
		truth_noalias += tmp_truth;
		ht_err_noalias += ht_err;
		htdl_err_noalias += htdl_err;
		//LOG(stdout, "valset_noalias %ld ", valset_noalias);
		//LOG(stdout, "truth_noalias %ld\n", truth_noalias);
		//memopd_num++;
	}

	LOG(stdout, "~~~~~~~~~~~~~~~~~~~Result of No Alias Pair (HT)~~~~~~~~~~~~~~~~~~~\n");
	//LOG(stdout, "Total Memory Operand Number is %ld\n", memopd_num);
	LOG(stdout, "Total Hypothesis Testing No Alias Pair is %lld\n", ht_noalias);
	LOG(stdout, "Total HT & DL Analysis No Alias Pair is %lld\n", hand_noalias);
	LOG(stdout, "Total HT + DL Analysis No Alias Pair is %lld\n", htdl_noalias);
	LOG(stdout, "Total Ground Truth No Alias Pair is %lld\n", truth_noalias);
	//LOG(stdout, "Total Error Number for Valset Analysis is %lld\n", hta_err_noalias);
	//LOG(stdout, "Total Error Number for Valset + DL Analysis is %lld\n", dl_err_noalias);
	LOG(stdout, "No Alias Pair Percent for HT is %lf\n", ((double)ht_noalias)/truth_noalias);
	LOG(stdout, "No Alias Pair Percent for HT & DL is %lf\n", ((double)hand_noalias)/truth_noalias);
	LOG(stdout, "No Alias Pair Percent for HT + DL is %lf\n", ((double)htdl_noalias)/truth_noalias);
	LOG(stdout, "No Alias Pair Error Percent for HT is %lf\n", ((double)ht_err_noalias)/truth_noalias);
	LOG(stdout, "No Alias Pair Error Percent for HT + DL is %lf\n", ((double)htdl_err_noalias)/truth_noalias);
}

bool truth_verify_noalias(re_list_t *exp, re_list_t *entry) {
	unsigned long addr_entry, addr_exp;
	size_t size_entry, size_exp;
	x86_op_t *opd;

	opd = GET_OPERAND(exp);
	addr_exp = GET_TRUE_ADDR(exp);
	size_exp = translate_datatype_to_byte(opd->datatype);

	opd = GET_OPERAND(entry);
	addr_entry = GET_TRUE_ADDR(entry);
	size_entry = translate_datatype_to_byte(opd->datatype);
	//LOG(stdout, "addr_exp 0x%lx, size_exp %d, addr_entry 0x%lx, size_entry %d\n",
	//	    addr_exp, size_exp, addr_entry, size_entry);

	return nooverlap_mem(addr_exp, size_exp, addr_entry, size_entry);
}

static bool is_effective_reg(re_list_t *entry) {
	bool result = false;	
	x86_op_t *opd;

	if ((entry->node_type == DefNode) ||
	    (entry->node_type == UseNode && 
	     CAST2_USE(entry->node)->usetype == Opd)) {
		opd = GET_OPERAND(entry);

		if (opd->type == op_register) {
			result = true;
		}
	}
	if (entry->node_type == UseNode &&
	    CAST2_USE(entry->node)->usetype != Opd) {
			result = true;
	}
	return result;

}

static bool is_effective_exp(re_list_t *entry) {
	bool result = false;	
	x86_op_t *opd;

	if ((entry->node_type == DefNode) ||
	    (entry->node_type == UseNode && CAST2_USE(entry->node)->usetype == Opd)) {
		opd = GET_OPERAND(entry);

		if (opd->type == op_expression) {
			result = true;
		}
	}
	return result;
}

// only used to get value of register
static unsigned long get_regvalue_from_pin(re_list_t *reg) {
	unsigned long value = 0x0;
	operand_val_t *regvals;
	re_list_t *instnode;
	unsigned int index, regindex, regnum = 0;

	instnode = find_inst_of_node(reg);
	index = CAST2_INST(instnode->node)->inst_index;
	
	regvals = &re_ds.oplog_list.opval_list[index];

	assert(is_effective_reg(reg));

	switch (reg->node_type) {
	case DefNode:
		regnum = CAST2_DEF(reg->node)->operand->data.reg.id;
		break;
	case UseNode:
		if (CAST2_USE(reg->node)->usetype == Index) {
			regnum = CAST2_USE(reg->node)->operand->data.expression.index.id;
		} else if (CAST2_USE(reg->node)->usetype == Base) {
			regnum = CAST2_USE(reg->node)->operand->data.expression.base.id;
		} else if (CAST2_USE(reg->node)->usetype == Opd) {
			regnum = CAST2_USE(reg->node)->operand->data.reg.id;
		} else {
			assert(0 && "Error Invocation");
		}
		break;
	}

	assert(regnum);

	for(regindex = 0; regindex < regvals->regnum; regindex++){
		if (regvals->regs[regindex].reg_num == regnum) {
			value = regvals->regs[regindex].val.dword;
			break;
		}
	}
	return value;
} 

static unsigned long truth_address(re_list_t *exp) {
	unsigned long address = 0x0;
	unsigned long base_addr, index_addr;
	re_list_t *instnode, *definst, *defnode;
	int inst_index, type;
	re_list_t *base, *index;
	x86_op_t *opd;

	// get all the registers and calculate them together
	base_addr = 0x0;
	index_addr = 0x0;
	
	opd = GET_OPERAND(exp);

	get_element_of_exp(exp, &index, &base);
	
	// there may exist define of such register
	if (index) {
		instnode = find_inst_of_node(index);
		inst_index = CAST2_INST(instnode->node)->inst_index;
		
		index_addr = get_regvalue_from_pin(index);

		defnode = find_prev_def_of_use(index, &type);
		if (defnode) {
			definst = find_inst_of_node(defnode);
			//print_node(instnode);
			//print_node(defnode);
			assert(instnode->id != definst->id);
		}
	}

	if (base) {
		instnode = find_inst_of_node(base);
		inst_index = CAST2_INST(instnode->node)->inst_index;

		base_addr = get_regvalue_from_pin(base);

		if (re_ds.instlist[inst_index].type == insn_leave) {
			// esp = ebp
			// directly get address from ebp in intel pin trace
			re_list_t *dst[NOPD], *src[NOPD];
			int nuse, ndef;
			obtain_inst_operand(instnode, src, dst, &nuse, &ndef);
			//print_node(src[1]);
			base_addr = get_regvalue_from_pin(src[1]);
		} else {
			defnode = find_prev_def_of_use(base, &type);
			if (!defnode) goto cal_address;
			definst = find_inst_of_node(defnode);
			//print_node(instnode);
			//print_node(defnode);
			if (instnode->id != definst->id) goto cal_address;
			if (defnode->id > exp->id) {
				if (re_ds.instlist[inst_index].type == insn_push) {
					base_addr -= ADDR_SIZE_IN_BYTE;
				} else if (re_ds.instlist[inst_index].type == insn_call) {
					base_addr -= ADDR_SIZE_IN_BYTE;
				}
			} else {
				// No need to do anything
				// push [esp+0x1C]
			}
		}
	}
	
cal_address:
	address = base_addr + index_addr * opd->data.expression.scale + 
		(int)(opd->data.expression.disp);

	if (op_with_gs_seg(opd)) {
		address += re_ds.coredata->corereg.gs_base;
	}

	//LOG(stdout, "Address of expression is 0x%lx\n", address);

	return address;
}


void calculate_truth_address() {

	re_list_t *entry;
	unsigned int *true_addr;
	list_for_each_entry_reverse(entry, &re_ds.head.list, list) {
		if (entry->node_type == InstNode) continue;
		// only traverse the effective expressions
		if (is_effective_exp(entry)) {
			if (entry->node_type == DefNode) {
				CAST2_DEF(entry->node)->true_addr = truth_address(entry);
			} else {
				CAST2_USE(entry->node)->true_addr = truth_address(entry);
			}
		}
	}
}

void find_access_related_address(unsigned long address) {
	re_list_t *entry;
	unsigned long addr_exp;
	LOG(stdout, "List All the instructions related with %lx\n", address);
	list_for_each_entry(entry, &re_ds.head.list, list) {
		if (entry->node_type == InstNode) continue;
		addr_exp = GET_TRUE_ADDR(entry);
		if (addr_exp == address) {
			print_info_of_current_inst(find_inst_of_node(entry));
		}
	}
}
